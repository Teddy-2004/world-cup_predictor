"""
Feature Module 1 — Team Form

Computes rolling performance windows for every team across all competitions.
Every feature is computed BEFORE the match date so there's zero data leakage.

Features produced (prefix: home_ / away_):
  form_wins_{n}          Win rate over last n matches (n=5,10,20)
  form_goals_for_{n}     Avg goals scored
  form_goals_against_{n} Avg goals conceded
  form_gd_{n}            Avg goal difference
  form_xg_for_{n}        Avg xG (attack strength)
  form_xg_against_{n}    Avg xGA (defensive weakness)
  form_clean_sheets_{n}  Clean sheet rate
  form_pts_{n}           Points-per-game (3/1/0)
  form_momentum          Exp-decay weighted points (recent = heavier)
  form_scoring_streak    Consecutive matches scored
  form_unbeaten_streak   Consecutive matches without defeat
"""

import numpy as np
import pandas as pd


WINDOWS = [5, 10, 20]
DECAY   = 0.85   # weight of each match relative to the previous one


def _points(winner: str, perspective: str) -> float:
    """Convert winner string to points from a team's perspective."""
    if winner == "DRAW":
        return 1.0
    if winner == "HOME_TEAM" and perspective == "home":
        return 3.0
    if winner == "AWAY_TEAM" and perspective == "away":
        return 3.0
    return 0.0


def build_team_match_log(matches: pd.DataFrame, xg: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape the matches table into a long format:
    one row per (team, match) from both home and away perspective.
    Merge in xG where available.
    """
    matches = matches.copy()
    matches["match_date"] = pd.to_datetime(matches["match_date"])

    # ── Home perspective ───────────────────────────────────────────────────
    home = matches.rename(columns={
        "home_team":  "team",
        "away_team":  "opponent",
        "home_goals": "goals_for",
        "away_goals": "goals_against",
    }).assign(
        venue_side="home",
        points=lambda df: df.apply(
            lambda r: _points(r["winner"], "home"), axis=1
        ),
        win=lambda df: (df["winner"] == "HOME_TEAM").astype(int),
        draw=lambda df: (df["winner"] == "DRAW").astype(int),
        loss=lambda df: (df["winner"] == "AWAY_TEAM").astype(int),
        clean_sheet=lambda df: (df["goals_against"] == 0).astype(int),
    )[["id","match_date","competition","stage","team","opponent","venue_side",
       "goals_for","goals_against","points","win","draw","loss","clean_sheet"]]

    # ── Away perspective ───────────────────────────────────────────────────
    away = matches.rename(columns={
        "away_team":  "team",
        "home_team":  "opponent",
        "away_goals": "goals_for",
        "home_goals": "goals_against",
    }).assign(
        venue_side="away",
        points=lambda df: df.apply(
            lambda r: _points(r["winner"], "away"), axis=1
        ),
        win=lambda df: (df["winner"] == "AWAY_TEAM").astype(int),
        draw=lambda df: (df["winner"] == "DRAW").astype(int),
        loss=lambda df: (df["winner"] == "HOME_TEAM").astype(int),
        clean_sheet=lambda df: (df["goals_against"] == 0).astype(int),
    )[["id","match_date","competition","stage","team","opponent","venue_side",
       "goals_for","goals_against","points","win","draw","loss","clean_sheet"]]

    log = pd.concat([home, away], ignore_index=True).sort_values(
        ["team", "match_date"]
    ).reset_index(drop=True)

    # ── Merge xG ──────────────────────────────────────────────────────────
    if not xg.empty:
        xg = xg.copy()
        xg["match_date"] = pd.to_datetime(xg["match_date"])

        xg_home = xg[["match_date","home_team","away_team","home_xg","away_xg"]].rename(
            columns={"home_team":"team","away_team":"opponent",
                     "home_xg":"xg_for","away_xg":"xg_against"}
        )
        xg_away = xg[["match_date","away_team","home_team","away_xg","home_xg"]].rename(
            columns={"away_team":"team","home_team":"opponent",
                     "away_xg":"xg_for","home_xg":"xg_against"}
        )
        xg_long = pd.concat([xg_home, xg_away], ignore_index=True)
        log = log.merge(xg_long, on=["match_date","team","opponent"], how="left")
    else:
        log["xg_for"]     = np.nan
        log["xg_against"] = np.nan

    return log


def _rolling_window(group: pd.DataFrame, n: int) -> pd.DataFrame:
    """Compute rolling stats over the last n matches (strictly before each row)."""
    shifted = group.shift(1)   # no leakage: exclude current match

    def roll(col):
        return shifted[col].rolling(n, min_periods=1).mean()

    return pd.DataFrame({
        f"form_wins_{n}":          shifted["win"].rolling(n, min_periods=1).mean(),
        f"form_goals_for_{n}":     roll("goals_for"),
        f"form_goals_against_{n}": roll("goals_against"),
        f"form_gd_{n}":            roll("goals_for") - roll("goals_against"),
        f"form_xg_for_{n}":        roll("xg_for"),
        f"form_xg_against_{n}":    roll("xg_against"),
        f"form_clean_sheets_{n}":  shifted["clean_sheet"].rolling(n, min_periods=1).mean(),
        f"form_pts_{n}":           roll("points"),
    }, index=group.index)


def _exp_momentum(group: pd.DataFrame, decay: float = DECAY) -> pd.Series:
    """
    Exponentially-weighted points per game.
    Most recent match has weight 1.0, previous has 0.85, then 0.85², etc.
    """
    pts = group["points"].shift(1)   # no leakage
    weights = pd.Series(
        [decay ** i for i in range(len(pts))[::-1]],
        index=pts.index
    )
    weighted = (pts * weights).fillna(0)
    weight_sum = weights.where(pts.notna(), 0)
    momentum = weighted.expanding().sum() / weight_sum.expanding().sum().replace(0, np.nan)
    return momentum.rename("form_momentum")


def _streaks(group: pd.DataFrame) -> pd.DataFrame:
    """Scoring streak and unbeaten streak (before each match)."""
    scoring   = []
    unbeaten  = []
    sc_streak = 0
    ub_streak = 0

    for _, row in group.iterrows():
        scoring.append(sc_streak)
        unbeaten.append(ub_streak)
        # update streaks AFTER recording pre-match value
        if row["goals_for"] > 0:
            sc_streak += 1
        else:
            sc_streak = 0
        if row["win"] or row["draw"]:
            ub_streak += 1
        else:
            ub_streak = 0

    return pd.DataFrame({
        "form_scoring_streak":   scoring,
        "form_unbeaten_streak":  unbeaten,
    }, index=group.index)


def compute_form_features(log: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all form computations to the team match log.
    Returns the log with all form_* columns added.
    """
    log = log.sort_values(["team", "match_date"]).copy()
    result_parts = []

    for team, group in log.groupby("team", sort=False):
        group = group.copy()

        # Rolling windows
        for n in WINDOWS:
            window_df = _rolling_window(group, n)
            for col in window_df.columns:
                group[col] = window_df[col].values

        # Momentum
        group["form_momentum"] = _exp_momentum(group).values

        # Streaks
        streak_df = _streaks(group)
        for col in streak_df.columns:
            group[col] = streak_df[col].values

        result_parts.append(group)

    return pd.concat(result_parts, ignore_index=True)


def get_team_form_on_date(
    form_log: pd.DataFrame, team: str, before_date: str
) -> dict:
    """
    Extract the form feature row for a team just before a given match date.
    Used at prediction time for a future match.
    """
    before_date = pd.to_datetime(before_date)
    team_rows = form_log[
        (form_log["team"] == team) &
        (form_log["match_date"] < before_date)
    ].sort_values("match_date")

    if team_rows.empty:
        return {}

    last = team_rows.iloc[-1]
    form_cols = [c for c in form_log.columns if c.startswith("form_")]
    return last[form_cols].to_dict()