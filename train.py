"""
WC2026 Predictor — Train

Orchestrates the full training pipeline:
  1. Load feature matrix from parquet
  2. Split into train / holdout (last World Cup = test set)
  3. Fit MatchPredictor ensemble (Poisson + XGBoost + NN + meta-learner)
  4. Evaluate on holdout
  5. Save all models to disk

Usage:
    python train.py                        # full training
    python train.py --no-neural            # skip NN (faster, no torch needed)
    python train.py --tune                 # Optuna hyperparameter search first
    python train.py --holdout-year 2022    # use WC2022 as test set
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    log_loss, accuracy_score, classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, PARQUET_DIR
from models.ensemble import MatchPredictor


MODEL_DIR = DATA_DIR / "trained_models"


# ── Holdout split ─────────────────────────────────────────────────────────

def make_holdout_split(
    df: pd.DataFrame,
    holdout_year: int = 2022,
    competition_filter: str = "WC",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train/test split strategy:
      - Test  : World Cup matches from holdout_year (simulates predicting a real tournament)
      - Train : everything else (all competitions, all other years)

    This is the most realistic evaluation — the model never sees WC2022 during
    training, then we check how well it would have predicted those matches.
    """
    if "match_date" in df.columns:
        df = df.copy()
        df["_year"] = pd.to_datetime(df["match_date"]).dt.year

        test_mask = (
            (df["_year"] == holdout_year) &
            (df.get("competition", pd.Series("", index=df.index)) == competition_filter)
        ) if "competition" in df.columns else (df["_year"] == holdout_year)

        test_df  = df[test_mask].drop(columns=["_year"])
        train_df = df[~test_mask].drop(columns=["_year"])
    else:
        # Fallback: 80/20 time-ordered split
        split    = int(0.8 * len(df))
        train_df = df.iloc[:split]
        test_df  = df.iloc[split:]

    return train_df, test_df


# ── Evaluation report ─────────────────────────────────────────────────────

def evaluate_on_holdout(
    predictor: MatchPredictor,
    test_df: pd.DataFrame,
    target_col: str = "result",
) -> dict:
    """
    Run predictions on holdout set and print a full evaluation report.
    """
    print("\n" + "="*55)
    print("HOLDOUT EVALUATION")
    print("="*55)

    test_df = test_df[test_df[target_col].notna()].copy()
    if len(test_df) == 0:
        print("No holdout matches found.")
        return {}

    y_true = test_df[target_col].astype(int).values

    # Get predictions
    preds_df = predictor.predict_batch(test_df)
    probs    = preds_df[["p_away","p_draw","p_home"]].values
    y_pred   = np.argmax(probs, axis=1)

    # ── Core metrics ──────────────────────────────────────────────────────
    ll       = log_loss(y_true, probs)
    acc      = accuracy_score(y_true, y_pred)

    # Baseline: always predict the most common outcome (home win)
    baseline_probs = np.tile([0.28, 0.26, 0.46], (len(y_true), 1))
    baseline_ll    = log_loss(y_true, baseline_probs)
    baseline_acc   = accuracy_score(y_true, np.ones_like(y_true) * 2)  # always home

    print(f"\n  Matches evaluated : {len(y_true)}")
    print(f"  Log-loss          : {ll:.4f}  (baseline: {baseline_ll:.4f})")
    print(f"  Accuracy          : {acc:.3f}  (baseline: {baseline_acc:.3f})")
    print(f"  Improvement over baseline: {baseline_ll - ll:+.4f} log-loss")

    # ── Per-class breakdown ───────────────────────────────────────────────
    print("\n  Classification report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["Away Win", "Draw", "Home Win"],
        digits=3,
    ))

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    print("  Confusion matrix (rows=true, cols=pred):")
    print(f"  {'':12} {'Away':>6} {'Draw':>6} {'Home':>6}")
    for i, lbl in enumerate(["Away Win", "Draw    ", "Home Win"]):
        print(f"  {lbl}  {cm[i,0]:>6} {cm[i,1]:>6} {cm[i,2]:>6}")

    # ── Calibration check ─────────────────────────────────────────────────
    print("\n  Probability calibration (mean predicted vs actual rate):")
    for cls_idx, cls_name in enumerate(["Away", "Draw", "Home"]):
        mean_pred = probs[:, cls_idx].mean()
        actual    = (y_true == cls_idx).mean()
        print(f"    {cls_name:5s}: predicted={mean_pred:.3f}  actual={actual:.3f}  "
              f"diff={mean_pred-actual:+.3f}")

    # ── Per-model comparison ──────────────────────────────────────────────
    print("\n  Per-model log-loss on holdout:")
    for src, cols in [
        ("Poisson",  ["poisson_p_away","poisson_p_draw","poisson_p_home"]),
        ("XGBoost",  ["xgb_p_away","xgb_p_draw","xgb_p_home"]),
        ("Neural",   ["nn_p_away","nn_p_draw","nn_p_home"]),
        ("Ensemble", ["p_away","p_draw","p_home"]),
    ]:
        if all(c in preds_df.columns for c in cols):
            p = preds_df[cols].fillna(1/3).values
            try:
                src_ll = log_loss(y_true, p)
                print(f"    {src:10s}: {src_ll:.4f}")
            except Exception:
                pass

    print("="*55)

    metrics = {
        "holdout_log_loss":     ll,
        "holdout_accuracy":     acc,
        "baseline_log_loss":    baseline_ll,
        "baseline_accuracy":    baseline_acc,
        "improvement_log_loss": baseline_ll - ll,
        "n_holdout_matches":    len(y_true),
    }
    return metrics


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WC2026 Model Trainer")
    parser.add_argument("--no-neural",      action="store_true",
                        help="Skip neural network (no torch required)")
    parser.add_argument("--tune",           action="store_true",
                        help="Run Optuna hyperparameter search before training")
    parser.add_argument("--holdout-year",   type=int, default=2022,
                        help="Year of World Cup to hold out for evaluation")
    parser.add_argument("--folds",          type=int, default=5,
                        help="Number of CV folds for stacking")
    parser.add_argument("--features-path",  type=str,
                        default=str(DATA_DIR / "features.parquet"),
                        help="Path to feature matrix parquet")
    args = parser.parse_args()

    print("WC2026 Match Predictor — Model Training")
    print(f"Feature matrix: {args.features_path}")

    # ── Load features ─────────────────────────────────────────────────────
    feat_path = Path(args.features_path)
    if not feat_path.exists():
        print(f"\nERROR: Feature matrix not found at {feat_path}")
        print("Run 'python features/assembler.py' first to build features.")
        sys.exit(1)

    print("Loading feature matrix...")
    df = pd.read_parquet(feat_path)
    df = df[df["result"].notna()].copy()
    print(f"Loaded {len(df)} labelled matches, {df.shape[1]} columns")

    # ── Train/holdout split ───────────────────────────────────────────────
    train_df, test_df = make_holdout_split(df, holdout_year=args.holdout_year)
    print(f"Train: {len(train_df)} matches | Holdout (WC{args.holdout_year}): {len(test_df)} matches")

    # ── Optional: hyperparameter tuning ───────────────────────────────────
    if args.tune:
        print("\nRunning hyperparameter search (XGBoost)...")
        from models.xgboost_model import XGBoostOutcomeModel
        tuner = XGBoostOutcomeModel()
        best_params = tuner.tune(train_df, n_trials=50)
        print(f"Best params: {best_params}")

    # ── Train ensemble ────────────────────────────────────────────────────
    predictor = MatchPredictor(
        n_folds=args.folds,
        use_neural=not args.no_neural,
    )
    predictor.fit(train_df)

    # ── Evaluate on holdout ───────────────────────────────────────────────
    holdout_metrics = evaluate_on_holdout(predictor, test_df)

    # ── Save models ───────────────────────────────────────────────────────
    predictor.save(MODEL_DIR)

    # Save combined metrics
    all_metrics = {**predictor.eval_metrics, **holdout_metrics}
    with open(MODEL_DIR / "eval_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nModels saved to: {MODEL_DIR}/")
    print("Next step: python simulate.py")


if __name__ == "__main__":
    main()
