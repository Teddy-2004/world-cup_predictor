"""
Feature Module 6 — Master Assembler

Joins all feature modules into a single training-ready DataFrame.
One row per historical match, with:
  - All home_ and away_ features (from form, ELO, squad, context, environment, market)
  - Key differentials (home − away for symmetric features)
  - Target variables: result (0=away win, 1=draw, 2=home win), home_goals, away_goals
  - Imputation (median for numeric, 'UNKNOWN' for categorical)
  - StandardScaler fit on training data

Usage:
    assembler = MatchFeatureAssembler(matches, xg, elo, fifa, squad, weather)
    train_df  = assembler.build_training_set()
    X, y      = assembler.get_Xy(train_df)
    pred_row  = assembler.build_prediction_row("France", "Germany", "2026-06-20", "MetLife Stadium")
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from features.form import build_team_match_log, compute_form_features, get_team_form_on_date
from features.elo_squad import (
    build_elo_lookup, build_fifa_lookup, build_squad_lookup,
    attach_elo_features, attach_fifa_features, attach_squad_features,
)
from features.context import attach_context_features
from features.environment import attach_environment_features
from features.market_psychology import attach_market_psychology_features, build_streak_log


# Columns to drop from final feature set (leakage or not useful for model)
DROP_COLS = [
    "id", "source", "fetched_at", "status",
    "home_goals_ht", "away_goals_ht",
    "penalties_home", "penalties_away",
    # Raw team name strings — encoded separately if needed
]

# Target columns (never used as features)
TARGET_COLS = ["result", "home_goals", "away_goals", "winner"]


class MatchFeatureAssembler:
    """
    End-to-end feature engineering pipeline.

    Parameters
    ----------
    matches    : raw matches table (from DB)
    xg_df      : match_stats table with xG data
    elo_df     : elo_ratings table
    fifa_df    : fifa_rankings table
    squad_df   : squad_values table
    weather_df : venue_weather table
    """

    def __init__(
        self,
        matches: pd.DataFrame,
        xg_df: pd.DataFrame,
        elo_df: pd.DataFrame,
        fifa_df: pd.DataFrame,
        squad_df: pd.DataFrame,
        weather_df: pd.DataFrame,
    ):
        self.raw_matches = matches.copy()
        self.xg_df       = xg_df
        self.elo_lookup  = build_elo_lookup(elo_df)
        self.fifa_lookup = build_fifa_lookup(fifa_df)
        self.squad_lookup = build_squad_lookup(squad_df)
        self.weather_df  = weather_df

        # Pre-compute team match log with form features
        print("Building team match log + form features...")
        log = build_team_match_log(matches, xg_df)
        self.form_log = compute_form_features(log)
        self.streak_log = build_streak_log(matches)

        self.scaler   : StandardScaler | None = None
        self.imputer  : SimpleImputer   | None = None
        self.feature_cols: list[str]           = []

    # ── Build full training dataset ────────────────────────────────────────

    def build_training_set(self) -> pd.DataFrame:
        """
        Build the complete training DataFrame.
        Applies all feature modules then adds targets.
        """
        print("Attaching ELO features...")
        df = attach_elo_features(self.raw_matches, self.elo_lookup)

        print("Attaching FIFA ranking features...")
        df = attach_fifa_features(df, self.fifa_lookup)

        print("Attaching squad quality features...")
        df = attach_squad_features(df, self.squad_lookup)

        print("Attaching match context features...")
        df = attach_context_features(df)

        print("Attaching environment features...")
        df = attach_environment_features(df, self.weather_df)

        print("Attaching market + psychology features...")
        df = attach_market_psychology_features(df)

        print("Attaching form features (per-match lookup)...")
        df = self._attach_form_to_matches(df)

        print("Engineering differentials...")
        df = self._add_differentials(df)

        print("Adding target variables...")
        df = self._add_targets(df)

        print("Encoding categoricals...")
        df = self._encode_categoricals(df)

        print(f"Training set complete: {len(df)} rows × {df.shape[1]} cols")
        return df

    def _attach_form_to_matches(self, df: pd.DataFrame) -> pd.DataFrame:
        """Look up the pre-match form snapshot for each team in each match."""
        home_form_rows = []
        away_form_rows = []

        for _, m in df.iterrows():
            date = m["match_date"]

            hf = get_team_form_on_date(self.form_log, m["home_team"], date)
            af = get_team_form_on_date(self.form_log, m["away_team"], date)

            home_form_rows.append({f"home_{k}": v for k, v in hf.items()})
            away_form_rows.append({f"away_{k}": v for k, v in af.items()})

        home_df = pd.DataFrame(home_form_rows, index=df.index)
        away_df = pd.DataFrame(away_form_rows, index=df.index)
        return pd.concat([df, home_df, away_df], axis=1)

    def _add_differentials(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For every symmetric pair (home_X, away_X), add diff_X = home_X − away_X.
        This lets the model directly learn from advantage gaps.
        """
        df = df.copy()
        home_cols = [c for c in df.columns if c.startswith("home_") and
                     c.replace("home_","away_") in df.columns]

        for hcol in home_cols:
            acol = hcol.replace("home_", "away_")
            feat = hcol.replace("home_", "")
            try:
                df[f"diff_{feat}"] = df[hcol] - df[acol]
            except TypeError:
                pass   # skip non-numeric pairs

        return df

    def _add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add result (2=home win, 1=draw, 0=away win) and raw goal counts."""
        df = df.copy()
        df["result"] = df["winner"].map({
            "HOME_TEAM": 2,
            "DRAW":      1,
            "AWAY_TEAM": 0,
        })
        # Keep goals as regression targets too (for Poisson model)
        df["home_goals_target"] = df["home_goals"]
        df["away_goals_target"] = df["away_goals"]
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode confederation columns; drop raw string columns."""
        df = df.copy()

        for col in ["confederation_home", "confederation_away"]:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                df = pd.concat([df, dummies], axis=1)
                df.drop(columns=[col], inplace=True)

        # Drop leakage / non-feature columns
        drop = [c for c in DROP_COLS if c in df.columns]
        drop += ["winner", "home_team", "away_team", "competition",
                 "season", "stage", "venue_city", "match_date"]
        df.drop(columns=[c for c in drop if c in df.columns], inplace=True)

        return df

    # ── Fit imputer + scaler on training data ─────────────────────────────

    def fit_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit median imputer and standard scaler on the training set.
        Returns the transformed DataFrame.
        """
        target_cols = [c for c in TARGET_COLS + ["home_goals_target","away_goals_target"]
                       if c in df.columns]
        X = df.drop(columns=target_cols)

        # Numeric only — drop any remaining object/bool columns
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        # Drop constant columns (imputer/scaler can't handle them)
        num_cols = [c for c in num_cols if X[c].nunique(dropna=False) > 1]
        self.feature_cols = num_cols

        self.imputer = SimpleImputer(strategy="median")
        X_imp = self.imputer.fit_transform(X[num_cols])

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_imp)

        # Use the exact num_cols list (length matches X_scaled width)
        result = pd.DataFrame(X_scaled, columns=num_cols, index=df.index)
        for col in target_cols:
            if col in df.columns:
                result[col] = df[col].values

        return result

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply fitted imputer + scaler to new data."""
        assert self.imputer and self.scaler, "Call fit_preprocessing first."
        X = df[self.feature_cols]
        return self.scaler.transform(self.imputer.transform(X))

    def get_Xy(
        self, df: pd.DataFrame, target: str = "result"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) arrays ready for model training."""
        df_proc = self.fit_preprocessing(df)
        feat_cols = [c for c in df_proc.columns if c not in
                     TARGET_COLS + ["home_goals_target","away_goals_target"]]
        X = df_proc[feat_cols].values
        y = df_proc[target].values
        self.feature_cols = feat_cols
        return X, y

    # ── Build a single prediction row ─────────────────────────────────────

    def build_prediction_row(
        self,
        home_team: str,
        away_team: str,
        match_date: str,
        venue: str,
        stage: str = "GROUP_STAGE",
        odds_home: float = np.nan,
        odds_draw: float = np.nan,
        odds_away: float = np.nan,
    ) -> pd.DataFrame:
        """
        Build a single feature row for a future (unseen) match.
        Suitable for passing directly to a trained model.
        """
        # Create a synthetic match row
        mock = pd.DataFrame([{
            "id":          "PRED_001",
            "competition": "WC",
            "season":      "2026",
            "stage":       stage,
            "match_date":  match_date,
            "home_team":   home_team,
            "away_team":   away_team,
            "home_goals":  np.nan,
            "away_goals":  np.nan,
            "winner":      None,
            "venue_city":  venue,
            "status":      "SCHEDULED",
            "odds_home":   odds_home,
            "odds_draw":   odds_draw,
            "odds_away":   odds_away,
        }])

        # Attach all feature modules
        mock = attach_elo_features(mock, self.elo_lookup)
        mock = attach_fifa_features(mock, self.fifa_lookup)
        mock = attach_squad_features(mock, self.squad_lookup)
        mock = attach_context_features(mock)
        mock = attach_environment_features(mock, self.weather_df)
        mock = attach_market_psychology_features(mock)
        mock = self._attach_form_to_matches(mock)
        mock = self._add_differentials(mock)
        mock = self._encode_categoricals(mock)

        return mock

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Path):
        """Pickle the fitted assembler (scaler, imputer, feature_cols)."""
        with open(path, "wb") as f:
            pickle.dump({
                "scaler":       self.scaler,
                "imputer":      self.imputer,
                "feature_cols": self.feature_cols,
            }, f)
        print(f"Assembler saved → {path}")

    def load(self, path: Path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.scaler       = state["scaler"]
        self.imputer      = state["imputer"]
        self.feature_cols = state["feature_cols"]
        print(f"Assembler loaded ← {path}")


# ── Convenience runner ────────────────────────────────────────────────────

def build_features(parquet_dir: Path, output_path: Path) -> pd.DataFrame:
    """
    Load all parquet files from the data collection step,
    run the full feature engineering pipeline, and save the result.

    Returns the training-ready DataFrame.
    """
    print("Loading data...")
    matches  = pd.read_parquet(parquet_dir / "matches.parquet")
    xg       = pd.read_parquet(parquet_dir / "match_stats.parquet")
    elo      = pd.read_parquet(parquet_dir / "elo_ratings.parquet")
    fifa     = pd.read_parquet(parquet_dir / "fifa_rankings.parquet")
    squad    = pd.read_parquet(parquet_dir / "squad_values.parquet")
    weather  = pd.read_parquet(parquet_dir / "venue_weather.parquet")

    print(f"Matches: {len(matches)}")

    assembler = MatchFeatureAssembler(matches, xg, elo, fifa, squad, weather)
    df = assembler.build_training_set()

    # Drop rows with no target (unfinished matches)
    df = df[df["result"].notna()].copy()

    df.to_parquet(output_path, index=False)
    print(f"\nFeature matrix saved → {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nSample feature columns:")
    feat_cols = [c for c in df.columns if c not in TARGET_COLS + ["home_goals_target","away_goals_target"]]
    for c in feat_cols[:20]:
        print(f"  {c}")
    print(f"  ... ({len(feat_cols)} features total)")

    return df


if __name__ == "__main__":
    from config import PARQUET_DIR, DATA_DIR
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    output = DATA_DIR / "features.parquet"
    df = build_features(PARQUET_DIR, output)