"""
Model 1 — Poisson Regression (Score Predictor)

Models the number of goals each team scores as an independent Poisson process.
λ_home = exp(intercept + attack_home + defence_away + home_advantage + venue_factors)
λ_away = exp(intercept + attack_away + defence_home + venue_factors)

From λ_home and λ_away we derive:
  - P(score = i:j) for all i,j ∈ [0,9]
  - P(home win), P(draw), P(away win)
  - Expected goals for each team

This model is interpretable and handles score prediction directly,
which feeds into exact-score markets and the tournament simulator.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.stats import poisson
from scipy.optimize import minimize
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


MAX_GOALS = 8   # upper bound for score grid


class PoissonModel:
    """
    Dual Poisson regression for match score prediction.

    Two separate models:
      home_model: predicts λ_home (expected goals for home team)
      away_model: predicts λ_away (expected goals for away team)

    Features used (subset of full feature matrix most relevant to scoring):
      - form_goals_for_10, form_goals_against_10
      - form_xg_for_10, form_xg_against_10
      - elo_diff
      - squad_value_diff
      - altitude_stress (suppresses goals at altitude)
      - heat_index_c (fatigue reduces scoring)
      - is_knockout (knockout matches tend to be tighter)
      - stage_weight
    """

    # Features that predict attacking output
    HOME_ATTACK_FEATURES = [
        "home_form_goals_for_10",
        "home_form_xg_for_10",
        "home_form_pts_10",
        "away_form_goals_against_10",
        "away_form_xg_against_10",
        "elo_diff",
        "squad_value_diff",
        "altitude_stress",
        "heat_index_c",
        "is_knockout",
        "stage_weight",
        "diff_form_goals_for_10",
        "diff_elo_home",
        "tournament_exp_diff",
    ]

    AWAY_ATTACK_FEATURES = [
        "away_form_goals_for_10",
        "away_form_xg_for_10",
        "away_form_pts_10",
        "home_form_goals_against_10",
        "home_form_xg_against_10",
        "elo_diff",           # negative = away team stronger
        "squad_value_diff",
        "altitude_stress",
        "heat_index_c",
        "is_knockout",
        "stage_weight",
        "diff_form_goals_for_10",
        "diff_elo_home",
        "tournament_exp_diff",
    ]

    def __init__(self, alpha: float = 0.1):
        """alpha: L2 regularisation strength for PoissonRegressor."""
        self.alpha = alpha
        self.home_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("poisson", PoissonRegressor(alpha=alpha, max_iter=300)),
        ])
        self.away_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("poisson", PoissonRegressor(alpha=alpha, max_iter=300)),
        ])
        self._home_cols: list[str] = []
        self._away_cols: list[str] = []

    # ── Training ───────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "PoissonModel":
        """
        Fit both Poisson models.
        df must contain home_goals_target, away_goals_target, and feature cols.
        """
        df = df[df["home_goals_target"].notna() & df["away_goals_target"].notna()].copy()

        # Select available features (some may be missing in df)
        self._home_cols = [c for c in self.HOME_ATTACK_FEATURES if c in df.columns]
        self._away_cols = [c for c in self.AWAY_ATTACK_FEATURES if c in df.columns]

        X_h = df[self._home_cols].fillna(df[self._home_cols].median())
        X_a = df[self._away_cols].fillna(df[self._away_cols].median())
        y_h = df["home_goals_target"].clip(0, MAX_GOALS)
        y_a = df["away_goals_target"].clip(0, MAX_GOALS)

        self.home_model.fit(X_h, y_h)
        self.away_model.fit(X_a, y_a)
        return self

    # ── Prediction ────────────────────────────────────────────────────────

    def predict_lambda(self, row: pd.DataFrame) -> tuple[float, float]:
        """Return (λ_home, λ_away) — expected goals for each team."""
        X_h = row[self._home_cols].fillna(0)
        X_a = row[self._away_cols].fillna(0)
        lam_h = float(self.home_model.predict(X_h)[0])
        lam_a = float(self.away_model.predict(X_a)[0])
        return max(lam_h, 0.05), max(lam_a, 0.05)

    def predict_score_matrix(
        self, row: pd.DataFrame, max_goals: int = MAX_GOALS
    ) -> np.ndarray:
        """
        Returns a (max_goals+1) × (max_goals+1) matrix where
        M[i,j] = P(home scores i, away scores j).
        """
        lam_h, lam_a = self.predict_lambda(row)
        home_probs = np.array([poisson.pmf(g, lam_h) for g in range(max_goals + 1)])
        away_probs = np.array([poisson.pmf(g, lam_a) for g in range(max_goals + 1)])
        return np.outer(home_probs, away_probs)

    def predict_outcome_probs(self, row: pd.DataFrame) -> dict:
        """
        Returns P(home win), P(draw), P(away win) from the score matrix.
        Also returns expected goals and most likely score.
        """
        M = self.predict_score_matrix(row)
        p_home = float(np.sum(np.tril(M, -1)))   # home_goals > away_goals
        p_draw = float(np.sum(np.diag(M)))
        p_away = float(np.sum(np.triu(M,  1)))

        lam_h, lam_a = self.predict_lambda(row)

        # Most likely scoreline
        idx = np.unravel_index(M.argmax(), M.shape)
        most_likely_score = f"{idx[0]}:{idx[1]}"

        return {
            "poisson_p_home": p_home,
            "poisson_p_draw": p_draw,
            "poisson_p_away": p_away,
            "poisson_xg_home": lam_h,
            "poisson_xg_away": lam_a,
            "poisson_most_likely_score": most_likely_score,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run predictions for a full DataFrame."""
        results = [self.predict_outcome_probs(df.iloc[[i]]) for i in range(len(df))]
        return pd.DataFrame(results, index=df.index)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "PoissonModel":
        with open(path, "rb") as f:
            return pickle.load(f)
