"""
Model 2 — XGBoost Classifier (Outcome Predictor)

Predicts P(home win), P(draw), P(away win) as a 3-class classification problem.
Uses the full 139-feature matrix — XGBoost handles missing values, non-linearity,
and feature interactions natively.

Key design decisions:
  - Multiclass softprob output (gives calibrated probabilities)
  - TimeSeriesSplit cross-validation (no future leakage)
  - Optuna hyperparameter search (fast, Bayesian)
  - SHAP values for feature importance (interpretability)
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


class XGBoostOutcomeModel:
    """
    XGBoost multiclass classifier for match outcome.

    Target encoding:
      0 = away win
      1 = draw
      2 = home win

    Outputs: probability vector [P(away), P(draw), P(home)]
    """

    DEFAULT_PARAMS = {
        "objective":        "multi:softprob",
        "num_class":        3,
        "n_estimators":     400,
        "max_depth":        5,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 3,
        "gamma":            0.1,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "use_label_encoder": False,
        "eval_metric":      "mlogloss",
        "random_state":     42,
        "n_jobs":           -1,
    }

    def __init__(self, params: dict = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model: xgb.XGBClassifier | None = None
        self.feature_cols: list[str] = []
        self.cv_scores: dict = {}
        self._calibrated: CalibratedClassifierCV | None = None

    # ── Training ───────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "result",
        calibrate: bool = True,
        n_cv_splits: int = 5,
    ) -> "XGBoostOutcomeModel":
        """
        Fit XGBoost on the feature DataFrame.

        Uses TimeSeriesSplit so validation sets are always in the future
        relative to training — mimics real prediction conditions.
        """
        df = df[df[target_col].notna()].copy()

        # Feature selection — exclude target and meta columns
        exclude = {target_col, "home_goals_target", "away_goals_target",
                   "result", "home_goals", "away_goals"}
        self.feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                             if c not in exclude]

        X = df[self.feature_cols].values
        y = df[target_col].astype(int).values

        # ── Cross-validation ───────────────────────────────────────────────
        print("Running TimeSeriesSplit cross-validation...")
        tscv = TimeSeriesSplit(n_splits=n_cv_splits)
        base = xgb.XGBClassifier(**self.params)

        cv_logloss = []
        oof_probs  = np.full((len(y), 3), np.nan)
        oof_mask   = np.zeros(len(y), dtype=bool)

        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X)):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            fold_model = xgb.XGBClassifier(**self.params)
            fold_model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )

            oof_probs[va_idx] = fold_model.predict_proba(X_va)
            oof_mask[va_idx] = True
            ll = log_loss(y_va, oof_probs[va_idx])
            cv_logloss.append(ll)
            print(f"  Fold {fold+1}: log-loss = {ll:.4f}")

        self.cv_scores = {
            "log_loss_mean": np.mean(cv_logloss),
            "log_loss_std":  np.std(cv_logloss),
            "oof_log_loss":  log_loss(y[oof_mask], oof_probs[oof_mask]),
        }
        print(f"CV log-loss: {self.cv_scores['log_loss_mean']:.4f} "
              f"± {self.cv_scores['log_loss_std']:.4f}")
        print(f"OOF log-loss: {self.cv_scores['oof_log_loss']:.4f}")

        # ── Full model ─────────────────────────────────────────────────────
        print("Fitting final model on full training set...")
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y, verbose=False)

        # ── Probability calibration (Platt scaling) ────────────────────────
        if calibrate:
            print("Calibrating probabilities (isotonic regression)...")
            self._calibrated = CalibratedClassifierCV(
                self.model, method="isotonic", cv=3
            )
            self._calibrated.fit(X, y)

        return self

    # ── Hyperparameter search (Optuna) ─────────────────────────────────────

    def tune(
        self,
        df: pd.DataFrame,
        target_col: str = "result",
        n_trials: int = 50,
    ) -> dict:
        """
        Bayesian hyperparameter search with Optuna.
        Returns best params dict.

        Install: pip install optuna
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            print("optuna not installed — skipping tuning. pip install optuna")
            return self.params

        df = df[df[target_col].notna()].copy()
        exclude = {target_col, "home_goals_target", "away_goals_target",
                   "result", "home_goals", "away_goals"}
        feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                     if c not in exclude]
        X = df[feat_cols].values
        y = df[target_col].astype(int).values
        tscv = TimeSeriesSplit(n_splits=4)

        def objective(trial):
            params = {
                "objective":        "multi:softprob",
                "num_class":        3,
                "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
                "max_depth":        trial.suggest_int("max_depth", 3, 8),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "eval_metric":      "mlogloss",
                "use_label_encoder": False,
                "random_state":     42,
                "n_jobs":           -1,
            }
            scores = []
            for tr_idx, va_idx in tscv.split(X):
                m = xgb.XGBClassifier(**params)
                m.fit(X[tr_idx], y[tr_idx], verbose=False)
                scores.append(log_loss(y[va_idx], m.predict_proba(X[va_idx])))
            return np.mean(scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_params
        best.update({
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "use_label_encoder": False,
            "random_state": 42,
            "n_jobs": -1,
        })
        print(f"Best log-loss: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
        self.params = best
        return best

    # ── Prediction ────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns (n, 3) probability matrix: [P(away), P(draw), P(home)]."""
        clf = self._calibrated if self._calibrated else self.model
        return clf.predict_proba(X)

    def predict_match(self, row: pd.DataFrame) -> dict:
        """Predict a single match from a feature row."""
        # Only use feature cols present in this row; fill missing with 0
        available = [c for c in self.feature_cols if c in row.columns]
        X = row.reindex(columns=self.feature_cols).fillna(0).values.reshape(1, -1)
        probs = self.predict_proba(X)[0]
        return {
            "xgb_p_away": float(probs[0]),
            "xgb_p_draw": float(probs[1]),
            "xgb_p_home": float(probs[2]),
            "xgb_prediction": ["away", "draw", "home"][np.argmax(probs)],
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.reindex(columns=self.feature_cols).fillna(0).values
        probs = self.predict_proba(X)
        return pd.DataFrame({
            "xgb_p_away": probs[:, 0],
            "xgb_p_draw": probs[:, 1],
            "xgb_p_home": probs[:, 2],
        }, index=df.index)

    # ── Feature importance ────────────────────────────────────────────────

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Return top N most important features by gain."""
        scores = self.model.get_booster().get_score(importance_type="gain")
        imp = pd.DataFrame({
            "feature":    list(scores.keys()),
            "importance": list(scores.values()),
        }).sort_values("importance", ascending=False).head(top_n)
        return imp

    def shap_values(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for interpretability.
        Requires: pip install shap
        """
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            X = df[self.feature_cols].fillna(0).values
            return explainer.shap_values(X)
        except ImportError:
            print("shap not installed. pip install shap")
            return None

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"XGBoost model saved → {path}")

    @staticmethod
    def load(path: Path) -> "XGBoostOutcomeModel":
        with open(path, "rb") as f:
            return pickle.load(f)
