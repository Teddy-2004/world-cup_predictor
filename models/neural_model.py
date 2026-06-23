"""
Model 3 — Neural Network with Entity Embeddings

Uses a tabular neural network where team identities are learned as dense
vector embeddings (16-dim), then concatenated with the numeric feature vector
and passed through fully-connected layers.

Why embeddings for teams?
  Team identity captures latent traits that numeric features don't fully
  express: playing style, mentality, historical tournament DNA.
  Brazil's 16-dim embedding learns to encode "flair + pressure handling".
  England's learns "underperforms expectations in knockouts".

Architecture:
  [team_home_emb(16)] + [team_away_emb(16)] + [numeric_features(N)]
        → FC(256) → BN → Dropout(0.3)
        → FC(128) → BN → Dropout(0.2)
        → FC(64)
        → FC(3) → Softmax  [P(away), P(draw), P(home)]

Requires: pip install torch
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not installed. NeuralModel will be disabled. pip install torch")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


class _EmbeddingNet(nn.Module if TORCH_AVAILABLE else object):
    """
    PyTorch module: entity-embedding net for match outcome.
    """
    def __init__(
        self,
        n_teams: int,
        n_numeric: int,
        emb_dim: int = 16,
        hidden: list[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        hidden = hidden or [256, 128, 64]

        self.home_emb = nn.Embedding(n_teams, emb_dim, padding_idx=0)
        self.away_emb = nn.Embedding(n_teams, emb_dim, padding_idx=0)

        in_dim  = 2 * emb_dim + n_numeric
        layers  = []
        for i, h in enumerate(hidden):
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout if i == 0 else dropout * 0.7),
            ]
            in_dim = h

        layers.append(nn.Linear(in_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, team_home, team_away, x_numeric):
        emb_h = self.home_emb(team_home)
        emb_a = self.away_emb(team_away)
        x     = torch.cat([emb_h, emb_a, x_numeric], dim=1)
        return self.net(x)   # raw logits; apply softmax at inference


class NeuralOutcomeModel:
    """
    Wrapper around _EmbeddingNet with sklearn-compatible interface.

    If PyTorch is not available, predict_match() returns NaN probs
    gracefully so the ensemble still works.
    """

    def __init__(
        self,
        emb_dim: int = 16,
        hidden: list[int] = None,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 80,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ):
        self.emb_dim     = emb_dim
        self.hidden      = hidden or [256, 128, 64]
        self.dropout     = dropout
        self.lr          = lr
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.weight_decay = weight_decay
        self.device      = device

        self.team_encoder  = LabelEncoder()
        self.feature_cols: list[str] = []
        self.net: Optional[object]   = None
        self.train_losses: list[float] = []
        self.val_losses:   list[float] = []
        self._n_teams = 0

    # ── Training ───────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "result",
    ) -> "NeuralOutcomeModel":

        if not TORCH_AVAILABLE:
            print("PyTorch not available — NeuralModel skipped.")
            return self

        df = df[df[target_col].notna()].copy()

        # Encode team identities
        all_teams = pd.concat([df["home_team_raw"], df["away_team_raw"]]).unique() \
            if "home_team_raw" in df.columns else \
            np.array(["UNKNOWN"])
        # Fall back: try to find team names in df
        for col in ["home_team", "away_team"]:
            if col in df.columns:
                all_teams = pd.concat([df.get("home_team", pd.Series()), df.get("away_team", pd.Series())]).dropna().unique()
                break

        self.team_encoder.fit(np.append(all_teams, ["UNKNOWN"]))
        self._n_teams = len(self.team_encoder.classes_) + 1   # +1 for padding

        # Numeric features
        exclude = {target_col, "home_goals_target", "away_goals_target",
                   "result", "home_goals", "away_goals",
                   "home_team", "away_team", "home_team_raw", "away_team_raw"}
        self.feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                             if c not in exclude]

        X_num = df[self.feature_cols].fillna(0).values.astype(np.float32)

        # Team indices (0 if unknown)
        def encode_team(series):
            return np.array([
                self.team_encoder.transform([t])[0]
                if t in self.team_encoder.classes_ else 0
                for t in series
            ], dtype=np.int64)

        team_home_col = "home_team" if "home_team" in df.columns else None
        team_away_col = "away_team" if "away_team" in df.columns else None

        if team_home_col and team_away_col:
            t_home = encode_team(df[team_home_col])
            t_away = encode_team(df[team_away_col])
        else:
            t_home = np.zeros(len(df), dtype=np.int64)
            t_away = np.zeros(len(df), dtype=np.int64)

        y = df[target_col].astype(int).values

        # Train/val split (time-ordered: last 15% as validation)
        split = int(0.85 * len(df))
        X_tr, X_va   = X_num[:split], X_num[split:]
        th_tr, th_va = t_home[:split], t_home[split:]
        ta_tr, ta_va = t_away[:split], t_away[split:]
        y_tr, y_va   = y[:split], y[split:]

        dev = torch.device(self.device)
        self.net = _EmbeddingNet(
            n_teams=self._n_teams,
            n_numeric=X_num.shape[1],
            emb_dim=self.emb_dim,
            hidden=self.hidden,
            dropout=self.dropout,
        ).to(dev)

        optimizer = optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss()

        # DataLoaders
        def make_loader(X, th, ta, y, shuffle=True):
            ds = TensorDataset(
                torch.tensor(th, device=dev),
                torch.tensor(ta, device=dev),
                torch.tensor(X,  device=dev),
                torch.tensor(y,  device=dev),
            )
            return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

        tr_loader = make_loader(X_tr, th_tr, ta_tr, y_tr)
        va_loader = make_loader(X_va, th_va, ta_va, y_va, shuffle=False)

        best_val = float("inf")
        best_state = None

        print(f"Training neural net for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            # Train
            self.net.train()
            tr_loss = 0.0
            for th_b, ta_b, x_b, y_b in tr_loader:
                optimizer.zero_grad()
                logits = self.net(th_b, ta_b, x_b)
                loss   = criterion(logits, y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                tr_loss += loss.item()
            scheduler.step()

            # Validate
            self.net.eval()
            va_loss = 0.0
            with torch.no_grad():
                for th_b, ta_b, x_b, y_b in va_loader:
                    logits  = self.net(th_b, ta_b, x_b)
                    va_loss += criterion(logits, y_b).item()

            tr_loss /= len(tr_loader)
            va_loss /= max(len(va_loader), 1)
            self.train_losses.append(tr_loss)
            self.val_losses.append(va_loss)

            if va_loss < best_val:
                best_val   = va_loss
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.epochs} — train={tr_loss:.4f}  val={va_loss:.4f}")

        # Restore best checkpoint
        if best_state:
            self.net.load_state_dict(best_state)

        print(f"Best val loss: {best_val:.4f}")
        return self

    # ── Prediction ────────────────────────────────────────────────────────

    def _predict_tensor(self, df: pd.DataFrame) -> np.ndarray:
        if not TORCH_AVAILABLE or self.net is None:
            return np.full((len(df), 3), np.nan)

        dev = torch.device(self.device)
        X   = df[self.feature_cols].fillna(0).values.astype(np.float32)

        def encode_team(col):
            if col not in df.columns:
                return np.zeros(len(df), dtype=np.int64)
            return np.array([
                self.team_encoder.transform([t])[0]
                if t in self.team_encoder.classes_ else 0
                for t in df[col]
            ], dtype=np.int64)

        t_h = encode_team("home_team")
        t_a = encode_team("away_team")

        self.net.eval()
        with torch.no_grad():
            logits = self.net(
                torch.tensor(t_h, device=dev),
                torch.tensor(t_a, device=dev),
                torch.tensor(X,   device=dev),
            )
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict_match(self, row: pd.DataFrame) -> dict:
        probs = self._predict_tensor(row)[0]
        return {
            "nn_p_away": float(probs[0]),
            "nn_p_draw": float(probs[1]),
            "nn_p_home": float(probs[2]),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        probs = self._predict_tensor(df)
        return pd.DataFrame({
            "nn_p_away": probs[:, 0],
            "nn_p_draw": probs[:, 1],
            "nn_p_home": probs[:, 2],
        }, index=df.index)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Path):
        state = {k: v.cpu() for k, v in self.net.state_dict().items()} if self.net else None
        with open(path, "wb") as f:
            pickle.dump({
                "state_dict":   state,
                "config": {
                    "emb_dim": self.emb_dim, "hidden": self.hidden,
                    "dropout": self.dropout, "_n_teams": self._n_teams,
                },
                "team_encoder":  self.team_encoder,
                "feature_cols":  self.feature_cols,
            }, f)

    @staticmethod
    def load(path: Path) -> "NeuralOutcomeModel":
        with open(path, "rb") as f:
            data = pickle.load(f)
        m = NeuralOutcomeModel(**{k: v for k, v in data["config"].items()
                                  if k != "_n_teams"})
        m._n_teams    = data["config"]["_n_teams"]
        m.team_encoder = data["team_encoder"]
        m.feature_cols = data["feature_cols"]
        if TORCH_AVAILABLE and data["state_dict"]:
            m.net = _EmbeddingNet(
                n_teams=m._n_teams,
                n_numeric=len(m.feature_cols),
                emb_dim=m.emb_dim,
                hidden=m.hidden,
            )
            m.net.load_state_dict(data["state_dict"])
            m.net.eval()
        return m
