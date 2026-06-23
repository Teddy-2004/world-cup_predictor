"""
Feature Module 2 — ELO, FIFA Rankings & Squad Quality

Features produced:
  elo_home / elo_away          ELO rating on match date
  elo_diff                     home_elo − away_elo
  elo_home_peak / away_peak    Max ELO ever reached (proxy for ceiling)
  elo_home_momentum            ELO change over last 90 days
  elo_away_momentum
  fifa_rank_home / away        FIFA rank just before match
  fifa_rank_diff               home − away (negative = home is higher ranked)
  squad_value_home / away      Total squad market value EUR
  squad_value_diff             Ratio: home / (home + away)  ∈ [0, 1]
  squad_depth_score            Value of squad outside top-11 / total value
"""

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# ELO features
# ══════════════════════════════════════════════════════════════════════════════

def build_elo_lookup(elo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare ELO data for fast date-based lookup.
    Input: elo_ratings table (team, rating_date, elo).
    Returns sorted copy.
    """
    df = elo_df.copy()
    df["rating_date"] = pd.to_datetime(df["rating_date"])
    return df.sort_values(["team", "rating_date"])


def get_elo_on_date(elo_lookup: pd.DataFrame, team: str, date) -> float:
    """
    Find the most recent ELO rating for a team strictly before `date`.
    Returns NaN if no data.
    """
    date = pd.to_datetime(date)
    rows = elo_lookup[
        (elo_lookup["team"] == team) &
        (elo_lookup["rating_date"] <= date)
    ]
    return float(rows["elo"].iloc[-1]) if not rows.empty else np.nan


def get_elo_peak(elo_lookup: pd.DataFrame, team: str, before_date) -> float:
    """Highest ELO the team has ever reached before this match."""
    before_date = pd.to_datetime(before_date)
    rows = elo_lookup[
        (elo_lookup["team"] == team) &
        (elo_lookup["rating_date"] <= before_date)
    ]
    return float(rows["elo"].max()) if not rows.empty else np.nan


def get_elo_momentum(
    elo_lookup: pd.DataFrame, team: str, date, window_days: int = 90
) -> float:
    """ELO change over the last `window_days` days (positive = improving)."""
    date  = pd.to_datetime(date)
    start = date - pd.Timedelta(days=window_days)
    recent = elo_lookup[
        (elo_lookup["team"] == team) &
        (elo_lookup["rating_date"] >= start) &
        (elo_lookup["rating_date"] <= date)
    ].sort_values("rating_date")

    if len(recent) < 2:
        return np.nan
    return float(recent["elo"].iloc[-1]) - float(recent["elo"].iloc[0])


def attach_elo_features(
    matches: pd.DataFrame, elo_lookup: pd.DataFrame
) -> pd.DataFrame:
    """
    For each match row, look up ELO for home and away team and compute diffs.
    """
    matches = matches.copy()
    matches["match_date"] = pd.to_datetime(matches["match_date"])

    rows = []
    for _, m in matches.iterrows():
        date  = m["match_date"]
        home  = m["home_team"]
        away  = m["away_team"]

        elo_h = get_elo_on_date(elo_lookup, home, date)
        elo_a = get_elo_on_date(elo_lookup, away, date)

        rows.append({
            "elo_home":          elo_h,
            "elo_away":          elo_a,
            "elo_diff":          elo_h - elo_a if not np.isnan(elo_h + elo_a) else np.nan,
            "elo_home_peak":     get_elo_peak(elo_lookup, home, date),
            "elo_away_peak":     get_elo_peak(elo_lookup, away, date),
            "elo_home_momentum": get_elo_momentum(elo_lookup, home, date),
            "elo_away_momentum": get_elo_momentum(elo_lookup, away, date),
        })

    elo_features = pd.DataFrame(rows, index=matches.index)
    return pd.concat([matches, elo_features], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# FIFA Rankings features
# ══════════════════════════════════════════════════════════════════════════════

def build_fifa_lookup(fifa_df: pd.DataFrame) -> pd.DataFrame:
    df = fifa_df.copy()
    df["ranking_date"] = pd.to_datetime(df["ranking_date"])
    return df.sort_values(["team", "ranking_date"])


def get_fifa_rank_on_date(
    fifa_lookup: pd.DataFrame, team: str, date
) -> tuple[float, float]:
    """Returns (rank, points) just before the match date."""
    date = pd.to_datetime(date)
    rows = fifa_lookup[
        (fifa_lookup["team"] == team) &
        (fifa_lookup["ranking_date"] <= date)
    ]
    if rows.empty:
        return np.nan, np.nan
    last = rows.iloc[-1]
    return float(last["rank"]), float(last.get("points", np.nan))


def attach_fifa_features(
    matches: pd.DataFrame, fifa_lookup: pd.DataFrame
) -> pd.DataFrame:
    matches = matches.copy()
    matches["match_date"] = pd.to_datetime(matches["match_date"])

    rows = []
    for _, m in matches.iterrows():
        rank_h, pts_h = get_fifa_rank_on_date(fifa_lookup, m["home_team"], m["match_date"])
        rank_a, pts_a = get_fifa_rank_on_date(fifa_lookup, m["away_team"], m["match_date"])

        rows.append({
            "fifa_rank_home":   rank_h,
            "fifa_rank_away":   rank_a,
            "fifa_rank_diff":   rank_h - rank_a,   # negative = home better ranked
            "fifa_pts_home":    pts_h,
            "fifa_pts_away":    pts_a,
            "fifa_pts_diff":    pts_h - pts_a,
        })

    return pd.concat([matches, pd.DataFrame(rows, index=matches.index)], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# Squad quality features
# ══════════════════════════════════════════════════════════════════════════════

def build_squad_lookup(squad_df: pd.DataFrame) -> pd.DataFrame:
    df = squad_df.copy()
    df["valuation_date"] = pd.to_datetime(df["valuation_date"])
    return df.sort_values(["team", "valuation_date"])


def get_squad_value_on_date(
    squad_lookup: pd.DataFrame, team: str, date
) -> dict:
    """Get most recent squad valuation before the match date."""
    date = pd.to_datetime(date)
    rows = squad_lookup[
        (squad_lookup["team"] == team) &
        (squad_lookup["valuation_date"] <= date)
    ]
    if rows.empty:
        return {"total_value_eur": np.nan, "avg_value_eur": np.nan}
    last = rows.iloc[-1]
    return {
        "total_value_eur": float(last.get("total_value_eur", np.nan)),
        "avg_value_eur":   float(last.get("avg_value_eur",   np.nan)),
    }


def attach_squad_features(
    matches: pd.DataFrame, squad_lookup: pd.DataFrame
) -> pd.DataFrame:
    matches = matches.copy()
    matches["match_date"] = pd.to_datetime(matches["match_date"])

    rows = []
    for _, m in matches.iterrows():
        sq_h = get_squad_value_on_date(squad_lookup, m["home_team"], m["match_date"])
        sq_a = get_squad_value_on_date(squad_lookup, m["away_team"], m["match_date"])

        val_h = sq_h["total_value_eur"]
        val_a = sq_a["total_value_eur"]
        total = val_h + val_a if not np.isnan(val_h + val_a) else np.nan

        rows.append({
            "squad_value_home":  val_h,
            "squad_value_away":  val_a,
            "squad_value_diff":  val_h - val_a,
            # Proportion of combined value held by home team ∈ [0,1]
            "squad_value_share": val_h / total if (total and total > 0) else np.nan,
            "squad_avg_home":    sq_h["avg_value_eur"],
            "squad_avg_away":    sq_a["avg_value_eur"],
        })

    return pd.concat([matches, pd.DataFrame(rows, index=matches.index)], axis=1)