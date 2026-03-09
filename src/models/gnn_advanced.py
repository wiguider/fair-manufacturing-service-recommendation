"""Advanced GNN-based recommendation models: UltraGCN.

UltraGCN (Mao et al., 2021) approximates infinite-layer graph convolution
without explicit message passing.  Instead of iterative neighbourhood
aggregation it optimises two complementary objectives:

    1. User-item interaction loss  -- weighted BCE over observed pairs,
       where the per-pair weight omega_ui is derived from the graph Laplacian
       (captures how "similar" user u and item i are relative to the full
       graph structure).

    2. Item-item constraint loss   -- pulls each positive item embedding
       towards the embeddings of its top-K most similar neighbour items
       (beta_ij), approximating the infinite-GCN smoothness signal without
       expensive iterative propagation.

Reference:
    Mao, K., Zhu, J., Xiao, X., Lu, B., Wang, Z., & He, X. (2021).
    UltraGCN: Ultra Simplification of Graph Convolutional Networks for
    Recommendation. CIKM 2021.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix, diags

from src.models.baselines import BaseRecommender
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Core PyTorch module
# ---------------------------------------------------------------------------


class UltraGCNModel(nn.Module):
    """Embedding table and dot-product scoring for UltraGCN.

    The model is intentionally thin: just two embedding matrices.  All
    graph-aware loss terms are computed in the training loop using the
    pre-computed omega and beta coefficients, following the UltraGCN paper
    which treats graph structure as fixed hyper-parameters rather than
    learnable layers.
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items

        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

        # Xavier uniform matches the init used by BPR and LightGCN in this repo
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def predict_all_items(self, user_id: int) -> np.ndarray:
        """Return dot-product scores for all items for one user."""
        with torch.no_grad():
            u = self.user_emb.weight[user_id]
            scores = (u @ self.item_emb.weight.T).cpu().numpy()
        return scores


# ---------------------------------------------------------------------------
# Recommender wrapper
# ---------------------------------------------------------------------------


class UltraGCNRecommender(BaseRecommender):
    """UltraGCN recommender for implicit collaborative filtering.

    Parameters
    ----------
    embedding_dim:
        Dimension of user and item embedding vectors.
    ii_topk:
        Number of item-item neighbours stored in the beta matrix for the
        constraint loss.  Larger values improve approximation quality but
        increase memory and per-batch computation.
    lambda_constraint:
        Weight of the item-item constraint loss term (lambda * L_C).
    w1, w2, w3, w4:
        Importance weights for the four sub-terms in the UltraGCN loss:
        w1 = positive (user, item) BCE term,
        w2 = negative (user, item) BCE term,
        w3 = positive item-item constraint BCE term,
        w4 = negative item-item constraint BCE term.
        Setting all to 1.0 reproduces standard equally-weighted loss.
    neg_sample_ratio:
        Number of negative user-item samples drawn per positive interaction
        in each training step.
    constraint_neg_ratio:
        Number of negative item-item samples used in the constraint loss
        per positive neighbour.
    lr:
        Learning rate for Adam optimiser.
    weight_decay:
        L2 regularisation coefficient applied to embedding parameters.
    batch_size:
        Mini-batch size (number of positive interactions per batch).
    num_epochs:
        Maximum number of training epochs.
    patience:
        Early-stopping patience (epochs without loss improvement).
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        ii_topk: int = 10,
        lambda_constraint: float = 1e-3,
        w1: float = 1.0,
        w2: float = 1.0,
        w3: float = 1.0,
        w4: float = 1.0,
        neg_sample_ratio: int = 1,
        constraint_neg_ratio: int = 1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 1024,
        num_epochs: int = 100,
        patience: int = 10,
        seed: int = 42,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.ii_topk = ii_topk
        self.lambda_constraint = lambda_constraint
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.neg_sample_ratio = neg_sample_ratio
        self.constraint_neg_ratio = constraint_neg_ratio
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.seed = seed

        self.model: UltraGCNModel | None = None
        self.num_items: int = 0
        self.num_users: int = 0

        # Pre-computed graph quantities populated during fit()
        # Maps (user_id, item_id) -> omega scalar
        self._omega: dict[tuple[int, int], float] = {}
        # Maps item_id -> list of (neighbour_item_id, beta_ij) sorted descending
        self._ii_neighbours: dict[int, list[tuple[int, float]]] = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Graph pre-computation helpers
    # ------------------------------------------------------------------

    def _build_interaction_matrix(
        self,
        interactions: pd.DataFrame,
        num_users: int,
        num_items: int,
    ) -> csr_matrix:
        """Build binary user-item interaction matrix R of shape (U, I)."""
        users = interactions["user_id"].values
        items = interactions["item_id"].values
        data = np.ones(len(users), dtype=np.float32)
        return csr_matrix((data, (users, items)), shape=(num_users, num_items))

    def _compute_omega(
        self,
        R: csr_matrix,
        interactions: pd.DataFrame,
    ) -> None:
        """Compute per-(user, item) interaction weight omega_ui.

        Following Eq. (9) of the UltraGCN paper:

            omega_ui = 1 / sqrt(d_u) / sqrt(d_i)

        where d_u = |N(u)| and d_i = |N(i)| are the user/item degrees in
        the bipartite interaction graph.  This is the (u, i) entry of the
        symmetrically normalised adjacency restricted to observed pairs.
        """
        user_deg = np.asarray(R.sum(axis=1)).flatten().astype(np.float32)
        item_deg = np.asarray(R.sum(axis=0)).flatten().astype(np.float32)

        # Avoid division by zero for cold-start nodes
        user_deg = np.where(user_deg > 0, user_deg, 1.0)
        item_deg = np.where(item_deg > 0, item_deg, 1.0)

        self._omega = {}
        for _, row in interactions.iterrows():
            u = int(row["user_id"])
            i = int(row["item_id"])
            self._omega[(u, i)] = float(
                1.0 / np.sqrt(user_deg[u]) / np.sqrt(item_deg[i])
            )

    def _compute_beta(self, R: csr_matrix) -> None:
        """Compute item-item cosine similarity and store top-K neighbours.

        beta_ij is the cosine similarity between item i and item j measured
        in the shared user interaction space (columns of R).  We compute it
        as:

            beta_ij = (R[:, i]^T R[:, j]) / (||R[:, i]|| * ||R[:, j]||)

        For memory efficiency the similarity matrix is produced in chunks
        and only the top-K neighbours per item are retained.
        """
        # R_T: (num_items, num_users) -- each row is one item's user vector
        R_T = R.T.astype(np.float32)
        num_items = R_T.shape[0]

        # Row (item) L2 norms
        norms = np.asarray(R_T.power(2).sum(axis=1)).flatten()
        norms = np.where(norms > 0, np.sqrt(norms), 1.0).astype(np.float32)

        # Symmetrically normalise each row in-place
        inv_norms = diags(1.0 / norms)
        R_norm = inv_norms @ R_T  # (num_items, num_users), sparse

        # Dense representation for efficient matmul (fine for manufacturing-scale data)
        R_norm_dense = np.asarray(R_norm.todense(), dtype=np.float32)

        self._ii_neighbours = {}
        k = min(self.ii_topk, max(num_items - 1, 1))
        chunk_size = min(512, num_items)

        for start in range(0, num_items, chunk_size):
            end = min(start + chunk_size, num_items)
            chunk = R_norm_dense[start:end]                # (chunk, num_users)
            sim_chunk = chunk @ R_norm_dense.T             # (chunk, num_items)

            # Zero out self-similarity on the diagonal of this chunk
            for local_idx in range(end - start):
                sim_chunk[local_idx, start + local_idx] = 0.0

            # argpartition gives the k largest without full sort
            top_k_indices = np.argpartition(sim_chunk, -k, axis=1)[:, -k:]

            for local_idx in range(end - start):
                item_id = start + local_idx
                nb_ids = top_k_indices[local_idx]
                nb_sims = sim_chunk[local_idx, nb_ids].tolist()
                pairs = sorted(
                    zip(nb_ids.tolist(), nb_sims),
                    key=lambda x: x[1],
                    reverse=True,
                )
                self._ii_neighbours[item_id] = pairs

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def _user_item_loss(
        self,
        user: torch.Tensor,
        pos_item: torch.Tensor,
        neg_items: torch.Tensor,
        omega_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted BCE loss on user-item pairs.

        Implements Eq. (1) from the paper:
            L_O = -sum_{(u,i) in O} [ w1 * omega_ui * log(sigma(e_u . e_i))
                                     + w2 * log(1 - sigma(e_u . e_j)) ]

        user      : (B,)
        pos_item  : (B,)
        neg_items : (B, neg_sample_ratio)
        omega_pos : (B,)  per-sample omega weight for the positive pair
        """
        u_emb = self.model.user_emb(user)         # (B, D)
        p_emb = self.model.item_emb(pos_item)     # (B, D)
        n_emb = self.model.item_emb(neg_items)    # (B, S, D)

        pos_score = (u_emb * p_emb).sum(dim=-1)  # (B,)
        neg_score = (u_emb.unsqueeze(1) * n_emb).sum(dim=-1)  # (B, S)

        pos_loss = self.w1 * omega_pos * torch.log(torch.sigmoid(pos_score) + 1e-8)
        neg_loss = self.w2 * torch.log(1.0 - torch.sigmoid(neg_score) + 1e-8).sum(dim=1)

        return -(pos_loss + neg_loss).mean()

    def _constraint_loss(
        self,
        pos_item: torch.Tensor,
        ii_pos_items: torch.Tensor,
        ii_neg_items: torch.Tensor,
        beta_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Item-item constraint loss.

        Implements Eq. (4) from the paper:
            L_C = -sum_i [ sum_{j in N(i)} w3 * beta_ij * log(sigma(e_i . e_j))
                          + w4 * sum_{j- sampled} log(1 - sigma(e_i . e_j-)) ]

        pos_item     : (B,)
        ii_pos_items : (B, K)   top-K neighbour items
        ii_neg_items : (B, C)   randomly sampled negative items
        beta_pos     : (B, K)   beta weights for positive item-item pairs
        """
        anchor  = self.model.item_emb(pos_item)         # (B, D)
        pos_nb  = self.model.item_emb(ii_pos_items)     # (B, K, D)
        neg_nb  = self.model.item_emb(ii_neg_items)     # (B, C, D)

        a_exp = anchor.unsqueeze(1)                     # (B, 1, D)
        pos_score = (a_exp * pos_nb).sum(dim=-1)        # (B, K)
        neg_score = (a_exp * neg_nb).sum(dim=-1)        # (B, C)

        pos_loss = self.w3 * beta_pos * torch.log(torch.sigmoid(pos_score) + 1e-8)
        neg_loss = self.w4 * torch.log(1.0 - torch.sigmoid(neg_score) + 1e-8).sum(dim=1)

        return -(pos_loss.sum(dim=1) + neg_loss).mean()

    # ------------------------------------------------------------------
    # Public interface required by BaseRecommender
    # ------------------------------------------------------------------

    def fit(self, train_interactions: pd.DataFrame, item_features: pd.DataFrame = None) -> None:
        """Train UltraGCN on implicit feedback interactions.

        Parameters
        ----------
        train_interactions:
            DataFrame with columns user_id, item_id (and optionally rating).
        item_features:
            Not used by UltraGCN but accepted to match the BaseRecommender
            interface (content information could extend the model in future).
        """
        set_seed(self.seed)

        self.num_users = int(train_interactions["user_id"].max()) + 1
        self.num_items = int(train_interactions["item_id"].max()) + 1

        # --- Pre-compute graph quantities ---
        R = self._build_interaction_matrix(
            train_interactions, self.num_users, self.num_items
        )

        print("  UltraGCN: computing omega (interaction weights)...")
        self._compute_omega(R, train_interactions)

        print(f"  UltraGCN: computing item-item beta (top-{self.ii_topk} neighbours)...")
        self._compute_beta(R)

        # --- Build positive-item sets per user for negative sampling ---
        user_pos: dict[int, set[int]] = {}
        for _, row in train_interactions.iterrows():
            user_pos.setdefault(int(row["user_id"]), set()).add(int(row["item_id"]))

        users_arr = train_interactions["user_id"].values.astype(np.int64)
        items_arr = train_interactions["item_id"].values.astype(np.int64)
        n = len(users_arr)

        # --- Initialise model and optimiser ---
        self.model = UltraGCNModel(
            self.num_users, self.num_items, self.embedding_dim
        ).to(self.device)

        optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        rng = np.random.RandomState(self.seed)
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            perm = rng.permutation(n)
            total_loss = 0.0
            n_batches = 0

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = perm[start:end]
                B = end - start

                b_users = users_arr[idx]   # (B,)
                b_pos   = items_arr[idx]   # (B,)

                # ---- omega weights for positive (user, item) pairs ----
                omega_vals = np.array(
                    [self._omega.get((int(u), int(i)), 1.0)
                     for u, i in zip(b_users, b_pos)],
                    dtype=np.float32,
                )

                # ---- User-item negative sampling ----
                neg_ui = np.empty((B, self.neg_sample_ratio), dtype=np.int64)
                for bi, u in enumerate(b_users):
                    for si in range(self.neg_sample_ratio):
                        ni = rng.randint(0, self.num_items)
                        while ni in user_pos.get(int(u), set()):
                            ni = rng.randint(0, self.num_items)
                        neg_ui[bi, si] = ni

                # ---- Item-item positive neighbours and beta weights ----
                k = self.ii_topk
                ii_pos  = np.zeros((B, k), dtype=np.int64)
                ii_beta = np.zeros((B, k), dtype=np.float32)
                for bi, item in enumerate(b_pos):
                    neighbours = self._ii_neighbours.get(int(item), [])
                    for ki, (nb_id, beta_val) in enumerate(neighbours[:k]):
                        ii_pos[bi, ki]  = nb_id
                        ii_beta[bi, ki] = max(float(beta_val), 0.0)

                # ---- Item-item negative sampling ----
                c = self.constraint_neg_ratio
                ii_neg = np.empty((B, c), dtype=np.int64)
                for bi in range(B):
                    pos_nbs = {
                        int(nb)
                        for nb, _ in self._ii_neighbours.get(int(b_pos[bi]), [])
                    }
                    pos_nbs.add(int(b_pos[bi]))
                    for ci in range(c):
                        ni = rng.randint(0, self.num_items)
                        while ni in pos_nbs:
                            ni = rng.randint(0, self.num_items)
                        ii_neg[bi, ci] = ni

                # ---- Move tensors to device ----
                t_users   = torch.LongTensor(b_users).to(self.device)
                t_pos     = torch.LongTensor(b_pos).to(self.device)
                t_neg_ui  = torch.LongTensor(neg_ui).to(self.device)
                t_omega   = torch.FloatTensor(omega_vals).to(self.device)
                t_ii_pos  = torch.LongTensor(ii_pos).to(self.device)
                t_ii_neg  = torch.LongTensor(ii_neg).to(self.device)
                t_ii_beta = torch.FloatTensor(ii_beta).to(self.device)

                # ---- Compute combined loss ----
                loss_ui = self._user_item_loss(t_users, t_pos, t_neg_ui, t_omega)
                loss_ii = self._constraint_loss(t_pos, t_ii_pos, t_ii_neg, t_ii_beta)

                # L2 regularisation on batch embeddings only (not the full table)
                reg = (
                    self.model.user_emb(t_users).norm(2).pow(2)
                    + self.model.item_emb(t_pos).norm(2).pow(2)
                    + self.model.item_emb(t_neg_ui.view(-1)).norm(2).pow(2)
                ) / B * self.weight_decay

                loss = loss_ui + self.lambda_constraint * loss_ii + reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss_ui.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"  UltraGCN early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 20 == 0:
                print(
                    f"  UltraGCN epoch {epoch + 1}/{self.num_epochs}, "
                    f"ui_loss: {avg_loss:.4f}"
                )

        self.model.eval()
        print(f"  UltraGCN training complete (best ui_loss: {best_loss:.4f})")

    def predict(self, user_id: int, candidate_items: np.ndarray) -> np.ndarray:
        """Return predicted relevance scores for candidate items.

        Parameters
        ----------
        user_id:
            Integer user index (must have been seen during fit).
        candidate_items:
            Array of integer item indices to score.

        Returns
        -------
        np.ndarray of shape (len(candidate_items),) with float32 scores.
        """
        if self.model is None:
            return np.zeros(len(candidate_items))
        all_scores = self.model.predict_all_items(user_id)
        return all_scores[candidate_items]
