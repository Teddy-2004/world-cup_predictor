"""
Model 4 — Stacking Meta-Learner + Full Ensemble

Combines Poisson, XGBoost, and Neural Network predictions using a
logistic regression meta-learner trained on out-of-fold (OOF) predictions.

Why stacking (not simple averaging)?
  - Each model has different strengths: Poisson knows goal distribution,
    XGBoost captures complex feature interactions, NN captures team identity.
  - OOF training prevents the meta-learner from overfitting to base models.
  - Learned weights reflect each model's actual reliability, not assumed equality.

Pipeline:
  1. Split training data into K folds (time-ordered)
  2. For each fold: train base models on train split, predict on validation
  3. Collect OOF predictions as meta-features
  4. Train LogisticRegression meta-learner on (oof_poisson, oof_xgb, oof_nn) → target
  5. At inference: run all 3 base models → feed to meta-learner → final probs
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    log_loss, brier_score_loss, accuracy_score,
    classification_report, confusion_matrix
)

from models.poisson_model  import PoissonModel
from models.xgboost_model  import XGBoostOutcomeModel
from models.neural_model   import NeuralOutcomeModel


class MatchPredictor:
    """
    Full stacking ensemble: Poisson + XGBoost + Neural → LogReg meta-learner.

    Usage:
        predictor = MatchPredictor()
        predictor.fit(train_df)
        probs = predictor.predict_match(home="France", away="Germany", ...)
        predictor.save(Path("models/"))
    """

    def __init__(self, n_folds: int = 5, use_neural: bool = True):
        self.n_folds    = n_folds
        self.use_neural = use_neural

        # Base models
        self.poisson = PoissonModel(alpha=0.1)
        self.xgb     = XGBoostOutcomeModel()
        self.nn      = NeuralOutcomeModel(epochs=60) if use_neural else None

        # Meta-learner
        self.meta: LogisticRegression | None = None
        self.meta_feature_names: list[str]   = []

        # Evaluation
        self.eval_metrics: dict = {}
        self.feature_importance_df: pd.DataFrame | None = None

    # ── Training ───────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, target_col: str = "result") -> "MatchPredictor":
        """
        Full stacking training pipeline.
        df: complete feature DataFrame from assembler.build_training_set()
        """
        df = df[df[target_col].notna()].sort_values("match_date").copy() \
            if "match_date" in df.columns else df[df[target_col].notna()].copy()

        y = df[target_col].astype(int).values
        n = len(df)
        print(f"Training on {n} matches...")

        # ── Step 1: Generate OOF predictions for meta-learner ──────────────
        tscv = TimeSeriesSplit(n_splits=self.n_folds)

        oof_poisson = np.full((n, 3), 1/3)
        oof_xgb     = np.full((n, 3), 1/3)
        oof_nn      = np.full((n, 3), 1/3)   # fallback uniform if NN disabled
        oof_mask    = np.zeros(n, dtype=bool)

        print(f"\nGenerating OOF predictions ({self.n_folds} folds)...")
        for fold, (tr_idx, va_idx) in enumerate(tscv.split(np.arange(n))):
            tr_df = df.iloc[tr_idx]
            va_df = df.iloc[va_idx]
            print(f"  Fold {fold+1}: train={len(tr_df)}, val={len(va_df)}")
            oof_mask[va_idx] = True

            # ── Poisson OOF ────────────────────────────────────────────────
            p_fold = PoissonModel()
            p_fold.fit(tr_df)
            poisson_preds = p_fold.predict_batch(va_df)
            oof_poisson[va_idx, 0] = poisson_preds["poisson_p_away"]
            oof_poisson[va_idx, 1] = poisson_preds["poisson_p_draw"]
            oof_poisson[va_idx, 2] = poisson_preds["poisson_p_home"]

            # ── XGBoost OOF ────────────────────────────────────────────────
            x_fold = XGBoostOutcomeModel()
            x_fold.fit(tr_df, n_cv_splits=2)   # inner CV within fold
            xgb_preds = x_fold.predict_batch(va_df)
            oof_xgb[va_idx, 0] = xgb_preds["xgb_p_away"]
            oof_xgb[va_idx, 1] = xgb_preds["xgb_p_draw"]
            oof_xgb[va_idx, 2] = xgb_preds["xgb_p_home"]

            # ── Neural OOF ────────────────────────────────────────────────
            if self.use_neural:
                n_fold = NeuralOutcomeModel(epochs=40)
                n_fold.fit(tr_df)
                nn_preds = n_fold.predict_batch(va_df)
                if not nn_preds["nn_p_away"].isna().all():
                    oof_nn[va_idx, 0] = nn_preds["nn_p_away"]
                    oof_nn[va_idx, 1] = nn_preds["nn_p_draw"]
                    oof_nn[va_idx, 2] = nn_preds["nn_p_home"]

        # OOF log-losses
        print(f"\nOOF log-loss:")
        print(f"  Poisson: {log_loss(y[oof_mask], oof_poisson[oof_mask]):.4f}")
        print(f"  XGBoost: {log_loss(y[oof_mask], oof_xgb[oof_mask]):.4f}")
        if self.use_neural:
            print(f"  Neural:  {log_loss(y[oof_mask], oof_nn[oof_mask]):.4f}")

        # ── Step 2: Train meta-learner on OOF predictions ──────────────────
        print("\nTraining meta-learner...")
        meta_features = self._build_meta_features(oof_poisson, oof_xgb, oof_nn)
        self.meta_feature_names = [
            "poisson_p_away", "poisson_p_draw", "poisson_p_home",
            "xgb_p_away",     "xgb_p_draw",     "xgb_p_home",
            "nn_p_away",      "nn_p_draw",       "nn_p_home",
        ]

        self.meta = LogisticRegression(
            C=1.0,
            max_iter=500,
            random_state=42,
        )
        self.meta.fit(meta_features[oof_mask], y[oof_mask])

        meta_oof_ll = log_loss(y[oof_mask], self.meta.predict_proba(meta_features[oof_mask]))
        print(f"Meta-learner OOF log-loss: {meta_oof_ll:.4f}")

        # ── Step 3: Refit base models on full training set ──────────────────
        print("\nRefitting base models on full training data...")
        self.poisson.fit(df)
        self.xgb.fit(df)
        if self.use_neural and self.nn:
            self.nn.fit(df)

        # ── Step 4: Store meta-learner weights (interpretability) ──────────
        self._log_meta_weights()

        # ── Step 5: Evaluate on full training set (sanity check) ──────────
        self.eval_metrics = self._evaluate(df, y, oof_poisson, oof_xgb, oof_nn, meta_features)
        self.feature_importance_df = self.xgb.feature_importance(top_n=20)

        print("\n=== Training complete ===")
        self._print_eval()
        return self

    def _build_meta_features(self, p, x, n) -> np.ndarray:
        return np.hstack([p, x, n])

    def _log_meta_weights(self):
        if self.meta is None:
            return
        coefs = self.meta.coef_
        print("\nMeta-learner weights (per outcome class):")
        classes = ["away_win", "draw", "home_win"]
        for i, cls in enumerate(classes):
            feats = dict(zip(self.meta_feature_names, coefs[i]))
            top   = sorted(feats.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            print(f"  {cls}: " + ", ".join(f"{k}={v:.3f}" for k,v in top))

    def _evaluate(self, df, y, oof_p, oof_x, oof_n, meta_feats) -> dict:
        meta_probs = self.meta.predict_proba(meta_feats)
        y_pred     = np.argmax(meta_probs, axis=1)

        # Brier score per class (one-vs-rest)
        brier = {}
        for cls_idx, cls_name in enumerate(["away", "draw", "home"]):
            y_bin     = (y == cls_idx).astype(int)
            brier[cls_name] = brier_score_loss(y_bin, meta_probs[:, cls_idx])

        return {
            "log_loss":      log_loss(y, meta_probs),
            "accuracy":      accuracy_score(y, y_pred),
            "brier_home":    brier["home"],
            "brier_draw":    brier["draw"],
            "brier_away":    brier["away"],
            "n_matches":     len(y),
            "trained_at":    datetime.utcnow().isoformat(),
        }

    def _print_eval(self):
        m = self.eval_metrics
        print(f"  Log-loss:  {m['log_loss']:.4f}")
        print(f"  Accuracy:  {m['accuracy']:.3f}")
        print(f"  Brier (home/draw/away): "
              f"{m['brier_home']:.4f} / {m['brier_draw']:.4f} / {m['brier_away']:.4f}")
        if self.feature_importance_df is not None:
            print("\nTop 10 features (XGBoost gain):")
            for _, row in self.feature_importance_df.head(10).iterrows():
                print(f"  {row['feature']:<40} {row['importance']:.1f}")

    # ── Prediction ────────────────────────────────────────────────────────

    def predict_match(self, feature_row: pd.DataFrame) -> dict:
        """
        Predict a single match from a feature row.
        Returns a rich dict with probs from all models + final ensemble.
        """
        # Base model predictions
        poisson_out = self.poisson.predict_outcome_probs(feature_row)
        xgb_out     = self.xgb.predict_match(feature_row)
        nn_out      = self.nn.predict_match(feature_row) if self.nn else {
            "nn_p_away": 1/3, "nn_p_draw": 1/3, "nn_p_home": 1/3
        }
        for key in ["nn_p_away", "nn_p_draw", "nn_p_home"]:
            if not np.isfinite(nn_out.get(key, np.nan)):
                nn_out[key] = 1/3

        # Meta-learner
        meta_input = np.array([[
            poisson_out["poisson_p_away"], poisson_out["poisson_p_draw"], poisson_out["poisson_p_home"],
            xgb_out["xgb_p_away"],         xgb_out["xgb_p_draw"],         xgb_out["xgb_p_home"],
            nn_out["nn_p_away"],            nn_out["nn_p_draw"],            nn_out["nn_p_home"],
        ]])

        final_probs = self.meta.predict_proba(meta_input)[0]

        result = {
            **poisson_out,
            **xgb_out,
            **nn_out,
            "p_away":       float(final_probs[0]),
            "p_draw":       float(final_probs[1]),
            "p_home":       float(final_probs[2]),
            "prediction":   ["away_win", "draw", "home_win"][np.argmax(final_probs)],
        }
        return result

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict all rows in a DataFrame."""
        poisson_df = self.poisson.predict_batch(df)
        xgb_df     = self.xgb.predict_batch(df)
        nn_df      = self.nn.predict_batch(df) if self.nn else pd.DataFrame(
            {"nn_p_away": 1/3, "nn_p_draw": 1/3, "nn_p_home": 1/3}, index=df.index
        )

        meta_input = np.column_stack([
            poisson_df[["poisson_p_away","poisson_p_draw","poisson_p_home"]].values,
            xgb_df[["xgb_p_away","xgb_p_draw","xgb_p_home"]].values,
            nn_df[["nn_p_away","nn_p_draw","nn_p_home"]].fillna(1/3).values,
        ])

        final = self.meta.predict_proba(meta_input)
        result = pd.DataFrame({
            "p_away": final[:, 0],
            "p_draw": final[:, 1],
            "p_home": final[:, 2],
        }, index=df.index)
        return pd.concat([poisson_df, xgb_df, nn_df, result], axis=1)

    # ── Backtesting ───────────────────────────────────────────────────────

    def backtest(
        self, df: pd.DataFrame, target_col: str = "result"
    ) -> pd.DataFrame:
        """
        Walk-forward backtest: train on past, evaluate on future.
        Returns a DataFrame with predictions and outcomes per match.
        """
        df = df.sort_values("match_date").reset_index(drop=True) \
            if "match_date" in df.columns else df.reset_index(drop=True)
        df = df[df[target_col].notna()].copy()
        n  = len(df)

        results = []
        min_train = int(0.5 * n)   # need at least 50% for first training

        print(f"Walk-forward backtest: {n - min_train} test matches")
        for i in range(min_train, n):
            tr_df = df.iloc[:i]
            va_df = df.iloc[[i]]

            bt = MatchPredictor(n_folds=3, use_neural=False)
            bt.fit(tr_df)

            pred = bt.predict_match(va_df)
            true = int(va_df[target_col].iloc[0])

            results.append({
                "match_idx":  i,
                "true_result": true,
                "p_away":      pred["p_away"],
                "p_draw":      pred["p_draw"],
                "p_home":      pred["p_home"],
                "prediction":  pred["prediction"],
                "correct":     int(np.argmax([pred["p_away"], pred["p_draw"], pred["p_home"]]) == true),
                "log_loss":    -np.log(max([pred["p_away"], pred["p_draw"], pred["p_home"]][true], 1e-7)),
            })

        return pd.DataFrame(results)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, model_dir: Path):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        self.poisson.save(model_dir / "poisson.pkl")
        self.xgb.save(model_dir    / "xgboost.pkl")
        if self.nn:
            self.nn.save(model_dir / "neural.pkl")

        with open(model_dir / "meta_learner.pkl", "wb") as f:
            pickle.dump({
                "meta":               self.meta,
                "meta_feature_names": self.meta_feature_names,
                "eval_metrics":       self.eval_metrics,
                "use_neural":         self.use_neural,
            }, f)

        with open(model_dir / "eval_metrics.json", "w") as f:
            json.dump(self.eval_metrics, f, indent=2)

        print(f"All models saved → {model_dir}/")

    @staticmethod
    def load(model_dir: Path) -> "MatchPredictor":
        model_dir = Path(model_dir)

        with open(model_dir / "meta_learner.pkl", "rb") as f:
            meta_data = pickle.load(f)

        predictor = MatchPredictor(use_neural=meta_data["use_neural"])
        predictor.poisson            = PoissonModel.load(model_dir / "poisson.pkl")
        predictor.xgb                = XGBoostOutcomeModel.load(model_dir / "xgboost.pkl")
        predictor.meta               = meta_data["meta"]
        predictor.meta_feature_names = meta_data["meta_feature_names"]
        predictor.eval_metrics       = meta_data["eval_metrics"]

        nn_path = model_dir / "neural.pkl"
        if predictor.use_neural and nn_path.exists():
            predictor.nn = NeuralOutcomeModel.load(nn_path)

        print(f"Ensemble loaded ← {model_dir}/")
        return predictor
