"""
Feature Module 4 — Environment & Venue

Features produced:
  altitude_m               Venue altitude in metres
  altitude_stress          Non-linear stress score (0=sea level → 1=extreme altitude)
  temp_match_c             Expected match temperature (°C)
  humidity_match           Expected humidity (%)
  heat_index               Felt temperature combining heat + humidity
  temp_delta_home          |match temp − home country avg temp in that month|
  temp_delta_away
  climate_shock_home       Combined temp + humidity adaptation burden (0–1)
  climate_shock_away
  is_indoor                1 if stadium has a roof (affects heat/humidity)
  precip_risk              Precipitation probability on match date
"""

import numpy as np
import pandas as pd


# ── Venue metadata ─────────────────────────────────────────────────────────

VENUE_META = {
    "MetLife Stadium":         {"altitude_m": 3,    "is_indoor": 0, "city": "East Rutherford"},
    "SoFi Stadium":            {"altitude_m": 82,   "is_indoor": 1, "city": "Los Angeles"},
    "AT&T Stadium":            {"altitude_m": 186,  "is_indoor": 1, "city": "Dallas"},
    "Levi's Stadium":          {"altitude_m": 15,   "is_indoor": 0, "city": "Santa Clara"},
    "Arrowhead Stadium":       {"altitude_m": 280,  "is_indoor": 0, "city": "Kansas City"},
    "NRG Stadium":             {"altitude_m": 13,   "is_indoor": 1, "city": "Houston"},
    "Hard Rock Stadium":       {"altitude_m": 2,    "is_indoor": 0, "city": "Miami"},
    "Lincoln Financial Field": {"altitude_m": 11,   "is_indoor": 0, "city": "Philadelphia"},
    "Gillette Stadium":        {"altitude_m": 46,   "is_indoor": 0, "city": "Foxborough"},
    "Mercedes-Benz Stadium":   {"altitude_m": 316,  "is_indoor": 1, "city": "Atlanta"},
    "Lumen Field":             {"altitude_m": 5,    "is_indoor": 0, "city": "Seattle"},
    "BMO Field":               {"altitude_m": 76,   "is_indoor": 0, "city": "Toronto"},
    "BC Place":                {"altitude_m": 10,   "is_indoor": 1, "city": "Vancouver"},
    "Estadio Azteca":          {"altitude_m": 2240, "is_indoor": 0, "city": "Mexico City"},
    "Estadio Akron":           {"altitude_m": 1566, "is_indoor": 0, "city": "Guadalajara"},
    "Estadio BBVA":            {"altitude_m": 537,  "is_indoor": 0, "city": "Monterrey"},
}

# Monthly average temperatures (°C) by home country — used for climate delta
# Sourced from WorldClim averages; key months are June (5) and July (6)
COUNTRY_MONTHLY_TEMP = {
    # (jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec)
    "Germany":       (0,  1,  5, 10, 14, 17, 19, 19, 15, 10,  5,  1),
    "France":        (4,  4,  8, 12, 16, 19, 22, 22, 18, 13,  8,  4),
    "Spain":         (6,  7, 11, 13, 17, 22, 25, 25, 21, 16, 10,  7),
    "Portugal":      (9, 10, 12, 14, 17, 20, 22, 22, 20, 16, 12, 10),
    "England":       (4,  4,  7, 10, 13, 16, 18, 18, 15, 11,  7,  4),
    "Netherlands":   (3,  3,  6, 10, 14, 17, 19, 19, 15, 11,  7,  4),
    "Belgium":       (3,  3,  7, 10, 14, 17, 19, 19, 15, 11,  7,  4),
    "Croatia":       (2,  4,  8, 13, 18, 22, 25, 25, 20, 14,  8,  3),
    "Brazil":        (26,26, 26, 24, 23, 21, 21, 22, 23, 24, 25, 26),
    "Argentina":     (24,23, 21, 17, 13, 10,  9, 11, 14, 17, 20, 23),
    "Colombia":      (18,19, 19, 19, 19, 18, 18, 19, 19, 19, 19, 18),
    "Ecuador":       (18,18, 18, 18, 18, 17, 17, 17, 18, 18, 18, 18),
    "Uruguay":       (23,22, 20, 16, 13, 10,  9, 11, 13, 16, 19, 22),
    "Venezuela":     (26,26, 27, 27, 27, 26, 26, 27, 27, 27, 27, 26),
    "United States": (0,  2,  7, 13, 18, 23, 25, 24, 20, 13,  7,  2),
    "Mexico":        (16,18, 20, 22, 23, 23, 22, 22, 21, 19, 17, 16),
    "Canada":        (-9,-7,  -1,  6, 12, 17, 20, 19, 14,  7,  1, -6),
    "Japan":         (5,  6, 10, 15, 19, 23, 27, 28, 24, 18, 12,  7),
    "South Korea":   (0,  2,  7, 14, 19, 23, 27, 28, 23, 16,  8,  2),
    "Iran":          (4,  6, 12, 18, 24, 29, 32, 31, 27, 20, 12,  6),
    "Saudi Arabia":  (14,16, 21, 26, 32, 35, 37, 36, 33, 27, 21, 16),
    "Australia":     (25,25, 23, 20, 16, 13, 12, 13, 16, 19, 22, 24),
    "Morocco":       (12,13, 15, 17, 20, 23, 26, 26, 23, 20, 15, 12),
    "Senegal":       (22,23, 24, 26, 28, 28, 27, 27, 28, 28, 26, 23),
    "Egypt":         (13,15, 18, 22, 26, 29, 30, 30, 28, 24, 19, 14),
    "Nigeria":       (26,28, 28, 28, 27, 26, 25, 25, 25, 26, 27, 26),
    "Ivory Coast":   (27,27, 28, 27, 27, 25, 24, 24, 25, 26, 27, 27),
    "Ghana":         (27,28, 28, 28, 27, 26, 25, 25, 25, 26, 27, 27),
    "Cameroon":      (26,27, 27, 26, 25, 23, 22, 22, 23, 24, 25, 25),
    "Uzbekistan":    (-1, 1,  7, 15, 21, 26, 28, 27, 21, 13,  6,  1),
    "Jordan":        (8, 10, 14, 19, 24, 27, 29, 29, 27, 23, 16, 10),
    "Iraq":          (9, 11, 16, 22, 28, 33, 36, 35, 31, 25, 17, 11),
    "New Zealand":   (18,18, 16, 13, 10,  8,  7,  8, 10, 12, 14, 16),
}

# Monthly average temperatures for WC venue cities (Jun/Jul focus)
VENUE_CITY_MONTHLY_TEMP = {
    "East Rutherford": (0, 2,  7, 13, 19, 24, 27, 26, 22, 15,  9,  2),
    "Los Angeles":     (14,15, 16, 18, 20, 22, 24, 25, 24, 21, 17, 14),
    "Dallas":          (7, 10, 15, 20, 25, 29, 31, 31, 27, 21, 14,  8),
    "Santa Clara":     (10,12, 13, 15, 17, 19, 20, 20, 20, 17, 13, 10),
    "Kansas City":     (-1, 1,  7, 14, 20, 25, 28, 27, 22, 15,  7,  1),
    "Houston":         (11,13, 18, 22, 27, 30, 32, 32, 29, 23, 17, 12),
    "Miami":           (19,20, 22, 24, 27, 28, 29, 29, 28, 26, 23, 20),
    "Philadelphia":    (1,  2,  7, 14, 19, 25, 27, 26, 22, 16,  9,  3),
    "Foxborough":      (-2,-1,  4, 10, 16, 21, 24, 23, 19, 13,  7,  1),
    "Atlanta":         (5,  7, 12, 17, 22, 26, 28, 27, 24, 18, 12,  7),
    "Seattle":         (4,  5,  8, 11, 14, 17, 20, 20, 17, 12,  8,  5),
    "Toronto":         (-4,-3,  2,  9, 15, 21, 24, 23, 18, 12,  5, -1),
    "Vancouver":       (3,  5,  7, 10, 13, 16, 19, 19, 16, 11,  6,  4),
    "Mexico City":     (13,15, 17, 18, 19, 18, 17, 17, 17, 16, 14, 13),
    "Guadalajara":     (17,19, 21, 23, 25, 22, 21, 21, 21, 19, 18, 17),
    "Monterrey":       (14,17, 21, 25, 28, 29, 29, 29, 27, 22, 17, 14),
}

VENUE_TO_CITY = {v: m["city"] for v, m in VENUE_META.items()}


# ── Altitude stress ────────────────────────────────────────────────────────

def altitude_stress_score(altitude_m: float) -> float:
    """
    Non-linear stress score based on physiological research:
    - Below 1000m: negligible effect (score ~0)
    - 1000–2000m:  moderate (0.1–0.4)
    - 2000–3000m:  significant (0.4–0.8)
    - Above 3000m: severe (0.8–1.0)
    Uses a sigmoid-like curve.
    """
    if np.isnan(altitude_m):
        return np.nan
    if altitude_m < 500:
        return 0.0
    # Logistic-shaped curve, inflection at 2000m
    score = 1 / (1 + np.exp(-0.003 * (altitude_m - 2000)))
    return round(min(score, 1.0), 4)


# ── Heat index ────────────────────────────────────────────────────────────

def heat_index(temp_c: float, humidity_pct: float) -> float:
    """
    Rothfusz heat index (°C).
    Meaningful only when temp > 26°C and humidity > 40%.
    """
    if np.isnan(temp_c) or np.isnan(humidity_pct):
        return np.nan
    if temp_c < 26 or humidity_pct < 40:
        return temp_c   # no meaningful heat index effect

    T = temp_c * 9/5 + 32   # convert to °F for Rothfusz formula
    RH = humidity_pct

    HI_F = (-42.379 +
             2.04901523 * T +
             10.14333127 * RH -
             0.22475541 * T * RH -
             6.83783e-3 * T**2 -
             5.481717e-2 * RH**2 +
             1.22874e-3 * T**2 * RH +
             8.5282e-4 * T * RH**2 -
             1.99e-6 * T**2 * RH**2)

    return round((HI_F - 32) * 5/9, 2)   # back to °C


# ── Climate shock ─────────────────────────────────────────────────────────

def climate_shock(
    team: str,
    venue_city: str,
    month: int,
) -> dict:
    """
    Measure how different the match climate is from the team's home climate
    in the same month. Returns temp_delta and a combined shock score.
    """
    home_temps  = COUNTRY_MONTHLY_TEMP.get(team)
    venue_temps = VENUE_CITY_MONTHLY_TEMP.get(venue_city)

    if home_temps is None or venue_temps is None:
        return {"temp_delta": np.nan, "climate_shock": np.nan}

    home_t  = home_temps[month - 1]
    venue_t = venue_temps[month - 1]
    delta   = abs(venue_t - home_t)

    # Normalise: 0 = identical, 1 = ≥25°C difference (extreme)
    shock = min(delta / 25.0, 1.0)
    return {"temp_delta": delta, "climate_shock": round(shock, 4)}


# ── Master attacher ───────────────────────────────────────────────────────

def attach_environment_features(
    matches: pd.DataFrame,
    weather_df: pd.DataFrame,
    venue_col: str = "venue_city",
) -> pd.DataFrame:
    """
    Attach all environment features to matches.
    weather_df: venue_weather table from the database.
    """
    matches = matches.copy()
    matches["match_date"] = pd.to_datetime(matches["match_date"])

    # Prepare weather lookup: (venue, date) → stats
    if not weather_df.empty:
        weather_df = weather_df.copy()
        weather_df["weather_date"] = pd.to_datetime(weather_df["weather_date"])
        w_lookup = weather_df.set_index(["venue", "weather_date"])
    else:
        w_lookup = None

    rows = []
    for _, m in matches.iterrows():
        date   = m["match_date"]
        venue  = m.get(venue_col, "")
        month  = date.month

        meta   = VENUE_META.get(venue, {})
        alt    = meta.get("altitude_m", np.nan)
        indoor = meta.get("is_indoor", 0)
        city   = VENUE_TO_CITY.get(venue, "")

        # Weather on match day
        temp, humidity, precip = np.nan, np.nan, np.nan
        if w_lookup is not None:
            try:
                wrow     = w_lookup.loc[(venue, date.normalize())]
                temp     = float(wrow["temp_mean_c"])
                humidity = float(wrow["humidity_pct"])
                precip   = float(wrow["precipitation_mm"])
            except (KeyError, TypeError):
                pass

        # Fall back to historical average for that city/month
        if np.isnan(temp):
            city_temps = VENUE_CITY_MONTHLY_TEMP.get(city)
            if city_temps:
                temp = city_temps[month - 1]

        hi    = heat_index(temp, humidity)
        alt_s = altitude_stress_score(alt)

        shock_h = climate_shock(m["home_team"], city, month)
        shock_a = climate_shock(m["away_team"], city, month)

        rows.append({
            "altitude_m":            alt,
            "altitude_stress":       alt_s,
            "is_indoor":             indoor,
            "temp_match_c":          temp,
            "humidity_match":        humidity,
            "precip_mm":             precip,
            "heat_index_c":          hi,
            "temp_delta_home":       shock_h["temp_delta"],
            "temp_delta_away":       shock_a["temp_delta"],
            "climate_shock_home":    shock_h["climate_shock"],
            "climate_shock_away":    shock_a["climate_shock"],
            # Differential: positive = home team less climate-shocked
            "climate_advantage":     (
                shock_a["climate_shock"] - shock_h["climate_shock"]
                if not np.isnan(shock_h["climate_shock"] + shock_a["climate_shock"])
                else np.nan
            ),
        })

    return pd.concat([matches, pd.DataFrame(rows, index=matches.index)], axis=1)