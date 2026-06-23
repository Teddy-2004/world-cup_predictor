"""
Feature Module 5 — Market Signals & Psychology

Features produced:
  odds_implied_home        Implied win probability from closing odds (home)
  odds_implied_draw        Implied draw probability
  odds_implied_away        Implied away win probability
  odds_home_value          Kelly-criterion edge: model_prob − implied_prob
  wc_appearances           Number of previous World Cup appearances
  wc_best_result           Best WC finish encoded numerically (winner=1, final=2, …)
  tournament_exp_score     Weighted experience score across all major tournaments
  penalty_win_rate         Historical shootout win rate
  penalty_appearances      Number of shootout appearances
  current_win_streak       Consecutive wins entering this match
  current_loss_streak      Consecutive losses (negative pressure)
  days_since_trophy        Days since last major trophy (recency of success)
"""

import numpy as np
import pandas as pd


# ── Tournament history (manually curated — public record) ─────────────────

WC_HISTORY = {
    # team: (appearances, best_finish_code)
    # best_finish: 1=winner, 2=final, 3=semi, 4=quarter, 5=r16, 6=groups, 7=never
    "Brazil":        (22, 1), "Germany":       (20, 1), "Italy":         (18, 1),
    "Argentina":     (18, 1), "France":        (16, 1), "England":       (16, 3),
    "Spain":         (16, 1), "Netherlands":   (13, 2), "Uruguay":       (14, 1),
    "Belgium":       (14, 3), "Portugal":      (9,  3), "Mexico":        (17, 4),
    "Croatia":       (8,  2), "Denmark":       (6,  4), "Sweden":        (12, 3),
    "Cameroon":      (8,  5), "South Korea":   (11, 3), "Japan":         (7,  5),
    "Morocco":       (6,  3), "Senegal":       (3,  4), "Nigeria":       (7,  5),
    "United States": (11, 4), "Switzerland":   (12, 4), "Australia":     (6,  5),
    "Ecuador":       (4,  5), "Colombia":      (7,  4), "Ghana":         (4,  4),
    "Ivory Coast":   (4,  6), "Algeria":       (4,  5), "Tunisia":       (6,  6),
    "Saudi Arabia":  (6,  5), "Iran":          (6,  6), "Costa Rica":    (6,  4),
    "Canada":        (2,  6), "Serbia":        (13, 3), "Poland":        (9,  3),
    "Austria":       (8,  3), "Hungary":       (9,  2), "Czech Republic":(9,  3),
    "Turkey":        (2,  3), "Paraguay":      (9,  4), "Chile":         (9,  3),
    "Bolivia":       (3,  6), "Peru":          (5,  4), "Venezuela":     (0,  7),
    "Egypt":         (3,  6), "South Africa":  (3,  6), "Cameroon":      (8,  5),
    "Iraq":          (1,  6), "Jordan":        (0,  7), "Uzbekistan":    (0,  7),
    "New Zealand":   (2,  6), "Albania":       (0,  7), "Slovenia":      (1,  6),
    "Slovakia":      (2,  6), "Panama":        (1,  6), "Jamaica":       (1,  6),
    "Qatar":         (1,  6),
}

# Major penalty shootout record (WC + major tournaments) — win/total appearances
PENALTY_RECORD = {
    "Germany":       (5, 6), "Argentina":     (4, 6), "Brazil":        (2, 6),
    "France":        (3, 5), "England":       (2, 8), "Spain":         (3, 4),
    "Netherlands":   (2, 4), "Italy":         (3, 5), "Portugal":      (4, 5),
    "Croatia":       (3, 4), "Denmark":       (3, 4), "Mexico":        (0, 5),
    "United States": (1, 2), "South Korea":   (2, 3), "Japan":         (1, 3),
    "Uruguay":       (2, 3), "Colombia":      (1, 2), "Belgium":       (1, 2),
    "Morocco":       (2, 3), "Senegal":       (0, 2),
    # Default for teams with no recorded shootout
}
DEFAULT_PENALTY = (1, 2)   # assume 50% if no history


# Encoded numeric best WC finish (lower = better)
FINISH_SCORE = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2, 6: 0.1, 7: 0.0}


def wc_experience_score(team: str) -> dict:
    apps, best = WC_HISTORY.get(team, (0, 7))
    finish_s   = FINISH_SCORE.get(best, 0.0)
    # Normalise appearances (max ~22 for Brazil)
    apps_s     = min(apps / 22, 1.0)
    # Combined: 60% finish quality, 40% experience breadth
    exp_score  = 0.6 * finish_s + 0.4 * apps_s
    return {
        "wc_appearances":        apps,
        "wc_best_finish_code":   best,
        "tournament_exp_score":  round(exp_score, 4),
    }


def penalty_features(team: str) -> dict:
    wins, total = PENALTY_RECORD.get(team, DEFAULT_PENALTY)
    return {
        "penalty_appearances": total,
        "penalty_win_rate":    round(wins / total, 4) if total > 0 else 0.5,
    }


# ── Odds → implied probabilities ─────────────────────────────────────────

def odds_to_implied_prob(
    odds_home: float,
    odds_draw: float,
    odds_away: float,
) -> dict:
    """
    Convert decimal odds to implied probabilities, removing the overround
    (bookmaker margin) so probabilities sum to 1.
    """
    if any(pd.isna(x) or x <= 1.0 for x in [odds_home, odds_draw, odds_away]):
        return {
            "odds_implied_home": np.nan,
            "odds_implied_draw": np.nan,
            "odds_implied_away": np.nan,
        }

    raw_h = 1 / odds_home
    raw_d = 1 / odds_draw
    raw_a = 1 / odds_away
    total = raw_h + raw_d + raw_a   # >1 due to overround

    return {
        "odds_implied_home": round(raw_h / total, 5),
        "odds_implied_draw": round(raw_d / total, 5),
        "odds_implied_away": round(raw_a / total, 5),
    }


# ── Pressure / streak features ────────────────────────────────────────────

def build_streak_log(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Build a long-format match log with pre-match win/loss streak per team.
    Same logic as form.py but focused only on streaks.
    """
    matches = matches.copy()
    matches["match_date"] = pd.to_datetime(matches["match_date"])

    frames = []
    for side, team_col, opp_col, result_col in [
        ("home", "home_team", "away_team", "winner"),
        ("away", "away_team", "home_team", "winner"),
    ]:
        sub = matches[[
            "id","match_date","competition",team_col,"winner"
        ]].rename(columns={team_col: "team"})
        sub["won"]  = sub.apply(
            lambda r: int(r["winner"] == f"{'HOME' if side=='home' else 'AWAY'}_TEAM"), axis=1
        )
        sub["lost"] = sub.apply(
            lambda r: int(r["winner"] == f"{'AWAY' if side=='home' else 'HOME'}_TEAM"), axis=1
        )
        frames.append(sub)

    log = pd.concat(frames, ignore_index=True).sort_values(["team","match_date"])

    streaks = []
    for team, grp in log.groupby("team"):
        win_s = 0
        loss_s = 0
        for _, row in grp.iterrows():
            streaks.append({
                "match_id": row["id"],
                "team":     team,
                "match_date": row["match_date"],
                "current_win_streak":  win_s,
                "current_loss_streak": loss_s,
            })
            if row["won"]:
                win_s += 1
                loss_s = 0
            elif row["lost"]:
                loss_s += 1
                win_s = 0
            else:
                win_s = 0
                loss_s = 0

    return pd.DataFrame(streaks)


def get_streak_on_date(
    streak_log: pd.DataFrame, team: str, before_date
) -> dict:
    before_date = pd.to_datetime(before_date)
    rows = streak_log[
        (streak_log["team"] == team) &
        (streak_log["match_date"] < before_date)
    ].sort_values("match_date")

    if rows.empty:
        return {"current_win_streak": 0, "current_loss_streak": 0}
    last = rows.iloc[-1]
    return {
        "current_win_streak":  int(last["current_win_streak"]),
        "current_loss_streak": int(last["current_loss_streak"]),
    }


# ── Master attacher ───────────────────────────────────────────────────────

def attach_market_psychology_features(
    matches: pd.DataFrame,
    odds_col_home: str = "odds_home",
    odds_col_draw: str = "odds_draw",
    odds_col_away: str = "odds_away",
) -> pd.DataFrame:
    """
    Attach market + psychology features to the matches DataFrame.
    Odds columns are optional — if missing, implied probs will be NaN.
    """
    matches = matches.copy()
    matches["match_date"] = pd.to_datetime(matches["match_date"])

    streak_log = build_streak_log(matches)

    rows = []
    for _, m in matches.iterrows():
        date  = m["match_date"]
        home  = m["home_team"]
        away  = m["away_team"]

        # Odds → probabilities
        odds_feats = odds_to_implied_prob(
            m.get(odds_col_home, np.nan),
            m.get(odds_col_draw, np.nan),
            m.get(odds_col_away, np.nan),
        )

        # Tournament experience
        exp_h = wc_experience_score(home)
        exp_a = wc_experience_score(away)

        # Penalty record
        pen_h = penalty_features(home)
        pen_a = penalty_features(away)

        # Streaks (pre-match)
        str_h = get_streak_on_date(streak_log, home, date)
        str_a = get_streak_on_date(streak_log, away, date)

        rows.append({
            **odds_feats,

            "wc_appearances_home":       exp_h["wc_appearances"],
            "wc_appearances_away":       exp_a["wc_appearances"],
            "wc_best_finish_home":       exp_h["wc_best_finish_code"],
            "wc_best_finish_away":       exp_a["wc_best_finish_code"],
            "tournament_exp_home":       exp_h["tournament_exp_score"],
            "tournament_exp_away":       exp_a["tournament_exp_score"],
            "tournament_exp_diff":       exp_h["tournament_exp_score"] - exp_a["tournament_exp_score"],

            "penalty_win_rate_home":     pen_h["penalty_win_rate"],
            "penalty_win_rate_away":     pen_a["penalty_win_rate"],
            "penalty_appearances_home":  pen_h["penalty_appearances"],
            "penalty_appearances_away":  pen_a["penalty_appearances"],

            "win_streak_home":           str_h["current_win_streak"],
            "loss_streak_home":          str_h["current_loss_streak"],
            "win_streak_away":           str_a["current_win_streak"],
            "loss_streak_away":          str_a["current_loss_streak"],
            "streak_advantage":          str_h["current_win_streak"] - str_a["current_win_streak"],
        })

    return pd.concat([matches, pd.DataFrame(rows, index=matches.index)], axis=1)