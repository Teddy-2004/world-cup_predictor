"""
WC2026 Predictor — Configuration
All URLs, API keys, venue metadata, and DB path live here.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
DB_PATH    = DATA_DIR / "wc2026.db"
RAW_DIR    = DATA_DIR / "raw"
PARQUET_DIR = DATA_DIR / "parquet"

for d in [DATA_DIR, RAW_DIR, PARQUET_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── football-data.org ──────────────────────────────────────────────────────
# Free tier: 10 req/min, historical international data
# Register at https://www.football-data.org/client/register
FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "YOUR_FREE_API_KEY_HERE")
FOOTBALL_DATA_BASE    = "https://api.football-data.org/v4"

# Competition IDs on football-data.org
COMPETITIONS = {
    "WC":   2000,   # FIFA World Cup
    "EC":   2018,   # UEFA Euro
    "CONC": 2013,   # CONCACAF Gold Cup
    "COPA": 2152,   # Copa America
    "ACN":  2022,   # Africa Cup of Nations
}

# ── ClubElo ───────────────────────────────────────────────────────────────
# Completely free, no key needed, CSV endpoint
CLUBELO_BASE = "http://api.clubelo.com"

# ── Open-Meteo ────────────────────────────────────────────────────────────
# Completely free weather + historical climate API, no key needed
OPEN_METEO_BASE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

# ── WC 2026 Venues with geo + altitude ────────────────────────────────────
WC2026_VENUES = {
    "MetLife Stadium":        {"city": "East Rutherford", "lat": 40.8135, "lon": -74.0744, "altitude_m": 3},
    "SoFi Stadium":           {"city": "Los Angeles",     "lat": 33.9535, "lon": -118.3392,"altitude_m": 82},
    "AT&T Stadium":           {"city": "Dallas",          "lat": 32.7473, "lon": -97.0945, "altitude_m": 186},
    "Levi's Stadium":         {"city": "Santa Clara",     "lat": 37.4032, "lon": -121.9698,"altitude_m": 15},
    "Arrowhead Stadium":      {"city": "Kansas City",     "lat": 39.0489, "lon": -94.4839, "altitude_m": 280},
    "NRG Stadium":            {"city": "Houston",         "lat": 29.6847, "lon": -95.4107, "altitude_m": 13},
    "Hard Rock Stadium":      {"city": "Miami",           "lat": 25.9580, "lon": -80.2389, "altitude_m": 2},
    "Lincoln Financial Field":{"city": "Philadelphia",    "lat": 39.9008, "lon": -75.1675, "altitude_m": 11},
    "Gillette Stadium":       {"city": "Foxborough",      "lat": 42.0909, "lon": -71.2643, "altitude_m": 46},
    "Mercedes-Benz Stadium":  {"city": "Atlanta",         "lat": 33.7553, "lon": -84.4006, "altitude_m": 316},
    "Lumen Field":            {"city": "Seattle",         "lat": 47.5952, "lon": -122.3316,"altitude_m": 5},
    "BMO Field":              {"city": "Toronto",         "lat": 43.6333, "lon": -79.4187, "altitude_m": 76},
    "BC Place":               {"city": "Vancouver",       "lat": 49.2767, "lon": -123.1118,"altitude_m": 10},
    "Estadio Azteca":         {"city": "Mexico City",     "lat": 19.3029, "lon": -99.1505, "altitude_m": 2240},
    "Estadio Akron":          {"city": "Guadalajara",     "lat": 20.6850, "lon": -103.4669,"altitude_m": 1566},
    "Estadio BBVA":           {"city": "Monterrey",       "lat": 25.6694, "lon": -100.2435,"altitude_m": 537},
}

# ── WC 2026 Qualified Teams (48 teams) ────────────────────────────────────
WC2026_TEAMS = [
    # UEFA (16)
    "Germany","France","Spain","Portugal","England","Netherlands",
    "Belgium","Croatia","Serbia","Austria","Switzerland","Denmark",
    "Slovakia","Slovenia","Albania","Turkey",
    # CONMEBOL (6)
    "Brazil","Argentina","Colombia","Ecuador","Uruguay","Venezuela",
    # CONCACAF (6)
    "United States","Mexico","Canada","Panama","Costa Rica","Jamaica",
    # CAF (9)
    "Morocco","Senegal","Egypt","Nigeria","South Africa","Ivory Coast",
    "Ghana","Cameroon","Tunisia",
    # AFC (8)
    "Japan","South Korea","Iran","Saudi Arabia","Australia",
    "Uzbekistan","Jordan","Iraq",
    # OFC (1)
    "New Zealand",
    # Intercontinental playoff spots (confirmed as most likely qualifiers)
    "Paraguay",   # South American playoff winner
    "Georgia",    # Already confirmed via UEFA
]

# ── Scraping headers ───────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; WC2026-Predictor/1.0; "
        "research project; contact: your@email.com)"
    )
}

# ── Rate limits (seconds between requests per source) ─────────────────────
RATE_LIMITS = {
    "football_data": 6.5,   # free tier: 10/min → 6s gap
    "clubelo":       2.0,
    "open_meteo":    1.0,
    "fbref":         4.0,   # be polite to FBref, they're sensitive to scraping
    "transfermarkt": 3.0,
    "fifa":          2.0,
}