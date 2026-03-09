"""Graph-based recommendation models: LightGCN."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import LGConv
from torch_geometric.utils import degree

from src.models.baselines import BaseRecommender


class LightGCNModel(nn.Module):
    """LightGCN (He et al., 2020) for collaborative filtering."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers

        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])

    def get_all_embeddings(self, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)

        all_embeddings = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            all_embeddings.append(x)

        # Average over layers
        stacked = torch.stack(all_embeddings, dim=0)
        final = stacked.mean(dim=0)

        user_final = final[: self.num_users]
        item_final = final[self.num_users:]
        return user_final, item_final

    def forward(self, edge_index, user, pos_item, neg_item):
        user_emb, item_emb = self.get_all_embeddings(edge_index)

        u = user_emb[user]
        p = item_emb[pos_item]
        n = item_emb[neg_item]

        pos_score = (u * p).sum(dim=-1)
        neg_score = (u * n).sum(dim=-1)
        return pos_score, neg_score

    def predict_scores(self, edge_index, user_id: int) -> np.ndarray:
        with torch.no_grad():
            user_emb, item_emb = self.get_all_embeddings(edge_index)
            u = user_emb[user_id]
            scores = (u @ item_emb.T).cpu().numpy()
        return scores


class LightGCNRecommender(BaseRecommender):
    def __init__(
        self,
        embedding_dim: int = 64,
        num_layers: int = 3,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        batch_size: int = 1024,
        num_epochs: int = 100,
        patience: int = 10,
        seed: int = 42,
    ):
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.seed = seed
        self.model = None
        self.edge_index = None
        self.num_items = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_edge_index(self, interactions: pd.DataFrame, num_users: int) -> torch.Tensor:
        """Build bipartite graph edge index. Item IDs are offset by num_users."""
        users = interactions["user_id"].values
        items = interactions["item_id"].values + num_users

        # Undirected: add both directions
        src = np.concatenate([users, items])
        dst = np.concatenate([items, users])
        edge_index = torch.LongTensor(np.stack([src, dst])).to(self.device)
        return edge_index

    def fit(self, train_interactions: pd.DataFrame, item_features: pd.DataFrame = None):
        num_users = train_interactions["user_id"].max() + 1
        self.num_items = train_interactions["item_id"].max() + 1

        self.edge_index = self._build_edge_index(train_interactions, num_users)
        self.model = LightGCNModel(
            num_users, self.num_items, self.embedding_dim, self.num_layers
        ).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Build user positive sets for negative sampling
        user_pos = {}
        for _, row in train_interactions.iterrows():
            user_pos.setdefault(int(row["user_id"]), set()).add(int(row["item_id"]))

        rng = np.random.RandomState(self.seed)
        users = train_interactions["user_id"].values
        pos_items = train_interactions["item_id"].values

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            perm = rng.permutation(len(users))

            total_loss = 0
            n_batches = 0

            for start in range(0, len(users), self.batch_size):
                end = min(start + self.batch_size, len(users))
                batch_idx = perm[start:end]

                batch_users = torch.LongTensor(users[batch_idx]).to(self.device)
                batch_pos = torch.LongTensor(pos_items[batch_idx]).to(self.device)

                # Negative sampling
                neg = []
                for u in users[batch_idx]:
                    n = rng.randint(0, self.num_items)
                    while n in user_pos.get(u, set()):
                        n = rng.randint(0, self.num_items)
                    neg.append(n)
                batch_neg = torch.LongTensor(neg).to(self.device)

                pos_score, neg_score = self.model(self.edge_index, batch_users, batch_pos, batch_neg)
                loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

                # L2 regularization on initial embeddings
                reg_loss = (
                    self.model.user_emb(batch_users).norm(2).pow(2)
                    + self.model.item_emb(batch_pos).norm(2).pow(2)
                    + self.model.item_emb(batch_neg).norm(2).pow(2)
                ) / len(batch_users) * self.weight_decay

                total = loss + reg_loss
                optimizer.zero_grad()
                total.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"  LightGCN early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 20 == 0:
                print(f"  LightGCN epoch {epoch + 1}/{self.num_epochs}, loss: {avg_loss:.4f}")

        self.model.eval()

    def predict(self, user_id: int, candidate_items: np.ndarray) -> np.ndarray:
        if self.model is None or self.edge_index is None:
            return np.zeros(len(candidate_items))
        scores = self.model.predict_scores(self.edge_index, user_id)
        return scores[candidate_items]
