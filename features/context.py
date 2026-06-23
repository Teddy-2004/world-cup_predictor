"""
Feature Module 3 — Match Context

Features produced:
  stage_weight             Numeric importance of the stage (group=1 → final=7)
  is_knockout              Binary: 1 if knockout round
  rest_days_home           Days since home team's last match
  rest_days_away           Days since away team's last match
  rest_advantage           rest_days_home − rest_days_away
  travel_km_home           km travelled to venue by home team
  travel_km_away           km travelled to venue by away team
  travel_advantage         travel_km_away − travel_km_home (positive = home less tired)
  h2h_win_rate_home        Home team win rate in last 10 H2H matches
  h2h_goal_diff            Avg goal diff (home − away) in H2H
  h2h_matches              Count of H2H matches available
  confederation_home       One-hot encoded confederation (UEFA/CONMEBOL/etc.)
  confederation_away
  conf_match               1 if same confederation (familiarity)
"""

import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2


# ── Stage encoding ─────────────────────────────────────────────────────────

STAGE_WEIGHTS = {
    "GROUP_STAGE":           1,
    "ROUND_OF_32":           2,   # new in WC2026
    "LAST_32":               2,
    "ROUND_OF_16":           3,
    "LAST_16":               3,
    "QUARTER_FINALS":        4,
    "SEMI_FINALS":           5,
    "THIRD_PLACE":           5,
    "FINAL":                 7,
    # Euro / Copa aliases
    "PLAYOFFS_ROUND_ONE":    2,
    "PLAYOFFS_SEMI_FINALS":  3,
    "PLAYOFFS_FINAL":        4,
}

CONFEDERATION_MAP = {
    # UEFA
    "Germany":"UEFA","France":"UEFA","Spain":"UEFA","Portugal":"UEFA",
    "England":"UEFA","Netherlands":"UEFA","Belgium":"UEFA","Croatia":"UEFA",
    "Serbia":"UEFA","Austria":"UEFA","Switzerland":"UEFA","Denmark":"UEFA",
    "Slovakia":"UEFA","Slovenia":"UEFA","Albania":"UEFA","Turkey":"UEFA",
    "Italy":"UEFA","Poland":"UEFA","Ukraine":"UEFA","Wales":"UEFA",
    "Scotland":"UEFA","Czech Republic":"UEFA","Hungary":"UEFA","Romania":"UEFA",
    "Greece":"UEFA","Sweden":"UEFA","Norway":"UEFA","Finland":"UEFA",
    "Russia":"UEFA","Bosnia and Herzegovina":"UEFA","Montenegro":"UEFA",
    "North Macedonia":"UEFA","Kosovo":"UEFA","Iceland":"UEFA","Georgia":"UEFA",
    # CONMEBOL
    "Brazil":"CONMEBOL","Argentina":"CONMEBOL","Colombia":"CONMEBOL",
    "Ecuador":"CONMEBOL","Uruguay":"CONMEBOL","Venezuela":"CONMEBOL",
    "Chile":"CONMEBOL","Peru":"CONMEBOL","Paraguay":"CONMEBOL",
    "Bolivia":"CONMEBOL",
    # CONCACAF
    "United States":"CONCACAF","Mexico":"CONCACAF","Canada":"CONCACAF",
    "Panama":"CONCACAF","Costa Rica":"CONCACAF","Jamaica":"CONCACAF",
    "Honduras":"CONCACAF","El Salvador":"CONCACAF","Haiti":"CONCACAF",
    "Trinidad and Tobago":"CONCACAF","Cuba":"CONCACAF",
    # CAF
    "Morocco":"CAF","Senegal":"CAF","Egypt":"CAF","Nigeria":"CAF",
    "South Africa":"CAF","Ivory Coast":"CAF","Ghana":"CAF",
    "Cameroon":"CAF","Tunisia":"CAF","Mali":"CAF","Algeria":"CAF",
    "Zimbabwe":"CAF","Tanzania":"CAF","Cape Verde":"CAF",
    "Equatorial Guinea":"CAF","Mozambique":"CAF","Guinea":"CAF",
    "Burkina Faso":"CAF","Zambia":"CAF","Uganda":"CAF",
    # AFC
    "Japan":"AFC","South Korea":"AFC","Iran":"AFC","Saudi Arabia":"AFC",
    "Australia":"AFC","Uzbekistan":"AFC","Jordan":"AFC","Iraq":"AFC",
    "Qatar":"AFC","China":"AFC","United Arab Emirates":"AFC",
    "Bahrain":"AFC","Kuwait":"AFC","Oman":"AFC","Thailand":"AFC",
    # OFC
    "New Zealand":"OFC","Fiji":"OFC","Papua New Guinea":"OFC",
}

# Approximate home country centroids (lat, lon) for travel calculation
COUNTRY_COORDS = {
    "Germany":       (51.2, 10.5), "France":        (46.2, 2.2),
    "Spain":         (40.4, -3.7), "Portugal":      (39.6, -8.0),
    "England":       (52.4, -1.8), "Netherlands":   (52.1, 5.3),
    "Belgium":       (50.5, 4.5),  "Croatia":       (45.1, 15.2),
    "Serbia":        (44.0, 21.0), "Austria":       (47.5, 14.6),
    "Switzerland":   (46.8, 8.2),  "Denmark":       (56.3, 9.5),
    "Slovakia":      (48.7, 19.7), "Slovenia":      (46.1, 14.8),
    "Albania":       (41.2, 20.2), "Turkey":        (38.9, 35.2),
    "Brazil":        (-14.2, -51.9),"Argentina":    (-38.4, -63.6),
    "Colombia":      (4.6, -74.1), "Ecuador":       (-1.8, -78.2),
    "Uruguay":       (-32.5, -55.8),"Venezuela":    (6.4, -66.6),
    "United States": (37.1, -95.7),"Mexico":        (23.6, -102.6),
    "Canada":        (56.1, -106.3),"Panama":       (8.5, -80.8),
    "Costa Rica":    (9.7, -83.8), "Jamaica":       (18.1, -77.3),
    "Morocco":       (31.8, -7.1), "Senegal":       (14.5, -14.5),
    "Egypt":         (26.8, 30.8), "Nigeria":       (9.1, 8.7),
    "South Africa":  (-30.6, 22.9),"Ivory Coast":  (7.5, -5.5),
    "Ghana":         (7.9, -1.0),  "Cameroon":      (3.8, 11.5),
    "Tunisia":       (33.9, 9.6),  "Japan":         (36.2, 138.3),
    "South Korea":   (35.9, 127.8),"Iran":          (32.4, 53.7),
    "Saudi Arabia":  (23.9, 45.1), "Australia":     (-25.3, 133.8),
    "Uzbekistan":    (41.4, 64.6), "Jordan":        (30.6, 36.2),
    "Iraq":          (33.2, 43.7), "New Zealand":   (-40.9, 174.9),
    "Qatar":         (25.4, 51.2),
}

# WC2026 venue coordinates (from config, duplicated here for self-containment)
VENUE_COORDS = {
    "MetLife Stadium":         (40.8135, -74.0744),
    "SoFi Stadium":            (33.9535, -118.3392),
    "AT&T Stadium":            (32.7473, -97.0945),
    "Levi's Stadium":          (37.4032, -121.9698),
    "Arrowhead Stadium":       (39.0489, -94.4839),
    "NRG Stadium":             (29.6847, -95.4107),
    "Hard Rock Stadium":       (25.9580, -80.2389),
    "Lincoln Financial Field": (39.9008, -75.1675),
    "Gillette Stadium":        (42.0909, -71.2643),
    "Mercedes-Benz Stadium":   (33.7553, -84.4006),
    "Lumen Field":             (47.5952, -122.3316),
    "BMO Field":               (43.6333, -79.4187),
    "BC Place":                (49.2767, -123.1118),
    "Estadio Azteca":          (19.3029, -99.1505),
    "Estadio Akron":           (20.6850, -103.4669),
    "Estadio BBVA":            (25.6694, -100.2435),
}


def _haversine(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def travel_km(team: str, venue: str) -> float:
    """Approximate travel distance for a team to a given venue."""
    if team not in COUNTRY_COORDS or venue not in VENUE_COORDS:
        return np.nan
    t_lat, t_lon = COUNTRY_COORDS[team]
    v_lat, v_lon = VENUE_COORDS[venue]
    return _haversine(t_lat, t_lon, v_lat, v_lon)


# ── Rest days ─────────────────────────────────────────────────────────────

def build_last_match_lookup(matches: pd.DataFrame) -> pd.DataFrame:
    """For each (team, match_date), find the date of that team's previous match."""
    matches = matches.copy()
    matches["match_date"] = pd.to_datetime(matches["match_date"])

    home = matches[["match_date","home_team"]].rename(columns={"home_team":"team"})
    away = matches[["match_date","away_team"]].rename(columns={"away_team":"team"})
    all_apps = pd.concat([home, away], ignore_index=True).drop_duplicates()
    all_apps = all_apps.sort_values(["team","match_date"])
    all_apps["prev_match_date"] = all_apps.groupby("team")["match_date"].shift(1)
    return all_apps


def get_rest_days(lookup: pd.DataFrame, team: str, date) -> float:
    date = pd.to_datetime(date)
    row = lookup[(lookup["team"] == team) & (lookup["match_date"] == date)]
    if row.empty or pd.isna(row["prev_match_date"].iloc[0]):
        return np.nan
    return (date - row["prev_match_date"].iloc[0]).days


# ── H2H history ──────────────────────────────────────────────────────────

def compute_h2h(
    matches: pd.DataFrame,
    team_a: str,
    team_b: str,
    before_date,
    n_matches: int = 10,
) -> dict:
    """
    Head-to-head record between two teams strictly before a date.
    Considers both home/away orientations.
    """
    before_date = pd.to_datetime(before_date)
    mask = (
        (
            ((matches["home_team"] == team_a) & (matches["away_team"] == team_b)) |
            ((matches["home_team"] == team_b) & (matches["away_team"] == team_a))
        ) &
        (matches["match_date"] < before_date)
    )
    h2h = matches[mask].sort_values("match_date").tail(n_matches)

    if h2h.empty:
        return {
            "h2h_matches":       0,
            "h2h_win_rate_home": np.nan,
            "h2h_goal_diff":     np.nan,
        }

    wins_a, draws, wins_b = 0, 0, 0
    gd_total = 0

    for _, r in h2h.iterrows():
        if r["home_team"] == team_a:
            gf, ga = r["home_goals"], r["away_goals"]
        else:
            gf, ga = r["away_goals"], r["home_goals"]

        gd_total += gf - ga
        w = r["winner"]
        if (w == "HOME_TEAM" and r["home_team"] == team_a) or \
           (w == "AWAY_TEAM" and r["away_team"] == team_a):
            wins_a += 1
        elif w == "DRAW":
            draws += 1
        else:
            wins_b += 1

    n = len(h2h)
    return {
        "h2h_matches":       n,
        "h2h_win_rate_home": wins_a / n,
        "h2h_goal_diff":     gd_total / n,
    }


# ── Stage encoding ────────────────────────────────────────────────────────

def encode_stage(stage: str) -> dict:
    weight = STAGE_WEIGHTS.get(stage, 1)
    return {
        "stage_weight":  weight,
        "is_knockout":   int(weight >= 2),
    }


# ── Master context attacher ───────────────────────────────────────────────

def attach_context_features(
    matches: pd.DataFrame,
    venue_col: str = "venue_city",
) -> pd.DataFrame:
    """
    Attach all context features to the matches DataFrame.
    venue_col: column holding the venue name (should match VENUE_COORDS keys).
    """
    matches = matches.copy()
    matches["match_date"] = pd.to_datetime(matches["match_date"])

    rest_lookup = build_last_match_lookup(matches)
    rows = []

    for _, m in matches.iterrows():
        date     = m["match_date"]
        home     = m["home_team"]
        away     = m["away_team"]
        venue    = m.get(venue_col, "")

        # Stage
        stage_feats = encode_stage(m.get("stage", "GROUP_STAGE"))

        # Rest
        rest_h = get_rest_days(rest_lookup, home, date)
        rest_a = get_rest_days(rest_lookup, away, date)

        # Travel
        tkm_h = travel_km(home, venue)
        tkm_a = travel_km(away, venue)

        # H2H
        h2h = compute_h2h(matches, home, away, before_date=date)

        # Confederation
        conf_h = CONFEDERATION_MAP.get(home, "UNKNOWN")
        conf_a = CONFEDERATION_MAP.get(away, "UNKNOWN")

        rows.append({
            **stage_feats,
            "rest_days_home":      rest_h,
            "rest_days_away":      rest_a,
            "rest_advantage":      (rest_h - rest_a) if not np.isnan(rest_h + rest_a) else np.nan,
            "travel_km_home":      tkm_h,
            "travel_km_away":      tkm_a,
            "travel_advantage":    (tkm_a - tkm_h) if not np.isnan(tkm_h + tkm_a) else np.nan,
            **h2h,
            "confederation_home":  conf_h,
            "confederation_away":  conf_a,
            "conf_match":          int(conf_h == conf_a),
        })

    return pd.concat([matches, pd.DataFrame(rows, index=matches.index)], axis=1)