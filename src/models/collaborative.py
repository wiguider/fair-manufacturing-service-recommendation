"""Collaborative filtering models: BPR and NeuMF."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.models.baselines import BaseRecommender


class BPRDataset(Dataset):
    """Triplet dataset for BPR training: (user, positive_item, negative_item)."""

    def __init__(self, interactions: pd.DataFrame, num_items: int, seed: int = 42):
        self.users = interactions["user_id"].values
        self.pos_items = interactions["item_id"].values
        self.num_items = num_items
        self.rng = np.random.RandomState(seed)

        # Build positive item set per user for negative sampling
        self.user_pos = {}
        for u, i in zip(self.users, self.pos_items):
            self.user_pos.setdefault(u, set()).add(i)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        pos = self.pos_items[idx]
        neg = self.rng.randint(0, self.num_items)
        while neg in self.user_pos.get(u, set()):
            neg = self.rng.randint(0, self.num_items)
        return torch.LongTensor([u]), torch.LongTensor([pos]), torch.LongTensor([neg])


class BPRModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, user, pos_item, neg_item):
        u = self.user_emb(user.squeeze())
        p = self.item_emb(pos_item.squeeze())
        n = self.item_emb(neg_item.squeeze())
        pos_score = (u * p).sum(dim=-1)
        neg_score = (u * n).sum(dim=-1)
        return pos_score, neg_score

    def predict_scores(self, user_id: int) -> np.ndarray:
        with torch.no_grad():
            u = self.user_emb.weight[user_id]
            scores = (u @ self.item_emb.weight.T).cpu().numpy()
        return scores


class BPRRecommender(BaseRecommender):
    def __init__(
        self,
        embedding_dim: int = 64,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        num_epochs: int = 100,
        patience: int = 10,
        seed: int = 42,
    ):
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.seed = seed
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train_interactions: pd.DataFrame, item_features: pd.DataFrame = None):
        num_users = train_interactions["user_id"].max() + 1
        num_items = train_interactions["item_id"].max() + 1

        self.model = BPRModel(num_users, num_items, self.embedding_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        dataset = BPRDataset(train_interactions, num_items, self.seed)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            for user, pos, neg in loader:
                user, pos, neg = user.to(self.device), pos.to(self.device), neg.to(self.device)
                pos_score, neg_score = self.model(user, pos, neg)
                loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"  BPR early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 20 == 0:
                print(f"  BPR epoch {epoch + 1}/{self.num_epochs}, loss: {avg_loss:.4f}")

        self.model.eval()

    def predict(self, user_id: int, candidate_items: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(candidate_items))
        scores = self.model.predict_scores(user_id)
        return scores[candidate_items]


class NeuMFModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        # GMF path
        self.gmf_user_emb = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_emb = nn.Embedding(num_items, embedding_dim)
        # MLP path
        self.mlp_user_emb = nn.Embedding(num_users, embedding_dim)
        self.mlp_item_emb = nn.Embedding(num_items, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, embedding_dim),
            nn.ReLU(),
        )
        self.output = nn.Linear(embedding_dim * 2, 1)

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, user, item):
        gmf = self.gmf_user_emb(user) * self.gmf_item_emb(item)
        mlp_input = torch.cat([self.mlp_user_emb(user), self.mlp_item_emb(item)], dim=-1)
        mlp_out = self.mlp(mlp_input)
        combined = torch.cat([gmf, mlp_out], dim=-1)
        return self.output(combined).squeeze(-1)


class NeuMFRecommender(BaseRecommender):
    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        lr: float = 0.001,
        batch_size: int = 256,
        num_epochs: int = 100,
        patience: int = 10,
        seed: int = 42,
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.seed = seed
        self.model = None
        self.num_items = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train_interactions: pd.DataFrame, item_features: pd.DataFrame = None):
        num_users = train_interactions["user_id"].max() + 1
        self.num_items = train_interactions["item_id"].max() + 1

        self.model = NeuMFModel(num_users, self.num_items, self.embedding_dim, self.hidden_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        # Build training data with negative sampling
        rng = np.random.RandomState(self.seed)
        pos_pairs = train_interactions[["user_id", "item_id"]].values
        user_pos = {}
        for u, i in pos_pairs:
            user_pos.setdefault(u, set()).add(i)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            # Negative sampling
            neg_items = []
            for u, _ in pos_pairs:
                ni = rng.randint(0, self.num_items)
                while ni in user_pos.get(u, set()):
                    ni = rng.randint(0, self.num_items)
                neg_items.append(ni)

            all_users = np.concatenate([pos_pairs[:, 0], pos_pairs[:, 0]])
            all_items = np.concatenate([pos_pairs[:, 1], np.array(neg_items)])
            all_labels = np.concatenate([np.ones(len(pos_pairs)), np.zeros(len(neg_items))])

            # Shuffle
            perm = rng.permutation(len(all_users))
            all_users, all_items, all_labels = all_users[perm], all_items[perm], all_labels[perm]

            total_loss = 0
            n_batches = 0
            for start in range(0, len(all_users), self.batch_size):
                end = start + self.batch_size
                u = torch.LongTensor(all_users[start:end]).to(self.device)
                i = torch.LongTensor(all_items[start:end]).to(self.device)
                y = torch.FloatTensor(all_labels[start:end]).to(self.device)

                pred = self.model(u, i)
                loss = criterion(pred, y)

                optimizer.zero_grad()
                loss.backward()
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
                    print(f"  NeuMF early stopping at epoch {epoch + 1}")
                    break

        self.model.eval()

    def predict(self, user_id: int, candidate_items: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(candidate_items))
        with torch.no_grad():
            u = torch.LongTensor([user_id] * len(candidate_items)).to(self.device)
            items = torch.LongTensor(candidate_items).to(self.device)
            scores = self.model(u, items).cpu().numpy()
        return scores
