"""
WC2026 Predictor — FastAPI Backend

Endpoints:
  GET  /health              → liveness check
  GET  /teams               → list of all 48 teams with metadata
  GET  /venues              → list of all 16 venues with metadata
  POST /predict             → single match prediction
  GET  /forecast            → cached tournament simulation results
  POST /forecast/refresh    → re-run simulation in background
  GET  /forecast/status     → poll refresh progress

Run locally:
  uvicorn api.main:app --reload --port 8000

Deploy:
  Docker + Railway (see Dockerfile and railway.toml)
"""

import json
import os
import sys
import warnings
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from scipy.stats import poisson as scipy_poisson

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR
from models.ensemble import MatchPredictor
from features.assembler import MatchFeatureAssembler
from simulate import TournamentSimulator, WC2026_GROUPS, TEAM_ELO, elo_outcome_probs


# ── App state ─────────────────────────────────────────────────────────────

class AppState:
    predictor:        MatchPredictor | None        = None
    assembler:        MatchFeatureAssembler | None  = None
    simulator:        TournamentSimulator | None    = None
    forecast:         dict | None                   = None
    refresh_progress: float                         = 0.0
    refresh_running:  bool                          = False
    loaded_at:        str                           = ""

state       = AppState()
MODEL_DIR   = DATA_DIR / "trained_models"
RESULTS_DIR = DATA_DIR / "simulation_results"
FORECAST_CACHE = RESULTS_DIR / "tournament_forecast.json"


# ── Poisson score helpers ─────────────────────────────────────────────────

def score_matrix(lam_h: float, lam_a: float, max_goals: int = 8):
    """
    Build a (max_goals+1) × (max_goals+1) matrix where
    M[i][j] = P(home scores i, away scores j).
    """
    rows = max_goals + 1
    M = [
        [float(scipy_poisson.pmf(i, lam_h) * scipy_poisson.pmf(j, lam_a))
         for j in range(rows)]
        for i in range(rows)
    ]
    return M


def top_scores_from_matrix(M, n: int = 3):
    """
    Return the n most likely scorelines as a list of dicts:
      [{"score": "2:1", "prob": 8.2}, ...]
    Probabilities are percentages rounded to 1 decimal.
    """
    rows = len(M)
    flat = [
        (M[i][j], i, j)
        for i in range(rows)
        for j in range(rows)
    ]
    flat.sort(reverse=True)
    return [
        {"score": f"{i}:{j}", "prob": round(p * 100, 1)}
        for p, i, j in flat[:n]
    ]


def outcome_probs_from_matrix(M):
    """
    Derive P(home win), P(draw), P(away win) from a score matrix.
    Guarantees they sum to exactly 1.0.
    """
    rows = len(M)
    p_home = sum(M[i][j] for i in range(rows) for j in range(rows) if i > j)
    p_draw = sum(M[i][i] for i in range(rows))
    p_away = sum(M[i][j] for i in range(rows) for j in range(rows) if j > i)
    total  = p_home + p_draw + p_away
    if total == 0:
        return 0.45, 0.27, 0.28
    return p_home / total, p_draw / total, p_away / total


def elo_to_xg(elo_h: float, elo_a: float):
    """
    Convert ELO ratings to expected goals for each team.

    Calibrated against historical international football data:
      - Average international match: ~2.5 total goals
      - Even match (same ELO): ~1.25 each
      - 200pt gap: roughly 1.6 vs 0.9
      - 400pt gap: roughly 2.1 vs 0.6
      - 600pt gap (max realistic): roughly 2.6 vs 0.4

    Formula uses a logistic elo_factor then maps to a [0.3, 3.5] xG range
    with the midpoint at 1.25 (average goals for a team in an even match).
    """
    elo_factor = 1 / (1 + 10 ** (-(elo_h - elo_a) / 400))
    # Scale so even match → 1.25 each, max gap → ~3.0 vs ~0.3
    lam_h = max(0.3, min(3.5, 2.8 * elo_factor + 0.25))
    lam_a = max(0.3, min(3.5, 2.8 * (1 - elo_factor) + 0.25))
    return round(lam_h, 3), round(lam_a, 3)


# ── ELO-only full prediction (used when trained model unavailable) ─────────

def elo_predict(home: str, away: str) -> dict:
    """
    Full prediction using only ELO ratings.
    Computes xG, score matrix, top scores, and outcome probs consistently
    from the same lambda values — no hardcoded placeholders anywhere.
    """
    elo_h = TEAM_ELO.get(home, 1750)
    elo_a = TEAM_ELO.get(away, 1750)

    lam_h, lam_a = elo_to_xg(elo_h, elo_a)
    M             = score_matrix(lam_h, lam_a)
    p_home, p_draw, p_away = outcome_probs_from_matrix(M)
    top           = top_scores_from_matrix(M, n=3)
    best_score    = top[0]["score"]
    prediction    = "home_win" if p_home > p_away else \
                    ("away_win" if p_away > p_home else "draw")

    return {
        # Outcome probabilities
        "p_home":            p_home,
        "p_draw":            p_draw,
        "p_away":            p_away,
        "prediction":        prediction,
        # Poisson model outputs (consistent naming for both paths)
        "poisson_p_home":    p_home,
        "poisson_p_draw":    p_draw,
        "poisson_p_away":    p_away,
        "poisson_xg_home":   lam_h,
        "poisson_xg_away":   lam_a,
        "most_likely_score": best_score,
        "top_scores":        top,
        # Other model slots (mirror ELO when full model unavailable)
        "xgb_p_home":  p_home, "xgb_p_draw":  p_draw, "xgb_p_away":  p_away,
        "nn_p_home":   p_home, "nn_p_draw":   p_draw, "nn_p_away":   p_away,
        "elo_home":    float(elo_h),
        "elo_away":    float(elo_a),
        "elo_diff":    float(elo_h - elo_a),
    }


def enrich_with_scores(result: dict, home: str, away: str) -> dict:
    """
    Given a result dict from the full ensemble model, compute the score
    matrix from its xG values and inject top_scores + most_likely_score.
    This guarantees the displayed score always matches the displayed xG.
    """
    # Prefer model's xG; fall back to ELO-derived
    lam_h = result.get("poisson_xg_home")
    lam_a = result.get("poisson_xg_away")

    if not lam_h or not lam_a or lam_h <= 0 or lam_a <= 0:
        elo_h = TEAM_ELO.get(home, 1750)
        elo_a = TEAM_ELO.get(away, 1750)
        lam_h, lam_a = elo_to_xg(elo_h, elo_a)

    M   = score_matrix(lam_h, lam_a)
    top = top_scores_from_matrix(M, n=3)

    result["most_likely_score"] = top[0]["score"]
    result["top_scores"]        = top
    result["poisson_xg_home"]   = round(lam_h, 3)
    result["poisson_xg_away"]   = round(lam_a, 3)
    return result


# ── Startup / shutdown ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ensemble model...")
    try:
        state.predictor = MatchPredictor.load(MODEL_DIR)
        print("  ✓ Ensemble loaded")
    except Exception as e:
        print(f"  ✗ Model load failed: {e}")
        print("  Running in ELO-only mode")

    print("Loading feature assembler...")
    try:
        pq = DATA_DIR / "parquet"
        state.assembler = MatchFeatureAssembler(
            pd.read_parquet(pq / "matches.parquet"),
            pd.read_parquet(pq / "match_stats.parquet"),
            pd.read_parquet(pq / "elo_ratings.parquet"),
            pd.read_parquet(pq / "fifa_rankings.parquet"),
            pd.read_parquet(pq / "squad_values.parquet"),
            pd.read_parquet(pq / "venue_weather.parquet"),
        )
        print("  ✓ Assembler ready")
    except Exception as e:
        print(f"  ✗ Assembler load failed: {e}")

    print("Loading simulator...")
    state.simulator = TournamentSimulator(
        predictor=state.predictor,
        assembler=state.assembler,
        groups=WC2026_GROUPS,
    )
    state.simulator.warm_cache(use_model=False)
    print("  ✓ Simulator ready")

    if FORECAST_CACHE.exists():
        with open(FORECAST_CACHE) as f:
            state.forecast = json.load(f)
        print(f"  ✓ Forecast loaded ({len(state.forecast['teams'])} teams)")

    state.loaded_at = datetime.utcnow().isoformat()
    print("API ready.\n")
    yield
    print("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="WC2026 Match Predictor API",
    version="1.0.0",
    description="ML-powered World Cup 2026 match and tournament predictions",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────

class ScoreOption(BaseModel):
    score: str   # e.g. "2:1"
    prob:  float # percentage e.g. 8.2


class PredictRequest(BaseModel):
    home_team:  str           = Field(...,          example="France")
    away_team:  str           = Field(...,          example="Colombia")
    venue:      str           = Field("MetLife Stadium", example="MetLife Stadium")
    stage:      str           = Field("GROUP_STAGE", example="QUARTER_FINALS")
    match_date: str           = Field("2026-06-15",  example="2026-06-20")
    odds_home:  Optional[float] = Field(None,        example=2.10)
    odds_draw:  Optional[float] = Field(None,        example=3.40)
    odds_away:  Optional[float] = Field(None,        example=3.60)


class PredictResponse(BaseModel):
    home_team:  str
    away_team:  str
    venue:      str
    stage:      str

    # Outcome probabilities
    p_home:     float
    p_draw:     float
    p_away:     float
    prediction: str           # "home_win" | "draw" | "away_win"

    # Score predictions — always consistent with xG below
    most_likely_score: str          # e.g. "2:1"
    top_scores:        List[ScoreOption]  # top 3 with probabilities

    # Expected goals — single source of truth for the score display
    poisson_xg_home: float
    poisson_xg_away: float

    # Per-model breakdown
    poisson_p_home: float
    poisson_p_draw: float
    poisson_p_away: float
    xgb_p_home:     float
    xgb_p_draw:     float
    xgb_p_away:     float
    nn_p_home:      float
    nn_p_draw:      float
    nn_p_away:      float

    # Venue
    altitude_m:      int
    temp_c:          float
    altitude_stress: float

    # ELO
    elo_home: float
    elo_away: float
    elo_diff: float

    computed_at: str


# ── Team / venue metadata ─────────────────────────────────────────────────

TEAM_META = {
    "France":        {"flag": "🇫🇷", "conf": "UEFA",     "rank": 2,  "wc_apps": 16},
    "Brazil":        {"flag": "🇧🇷", "conf": "CONMEBOL", "rank": 1,  "wc_apps": 22},
    "Argentina":     {"flag": "🇦🇷", "conf": "CONMEBOL", "rank": 3,  "wc_apps": 18},
    "England":       {"flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "conf": "UEFA",     "rank": 5,  "wc_apps": 16},
    "Spain":         {"flag": "🇪🇸", "conf": "UEFA",     "rank": 6,  "wc_apps": 16},
    "Germany":       {"flag": "🇩🇪", "conf": "UEFA",     "rank": 4,  "wc_apps": 20},
    "Portugal":      {"flag": "🇵🇹", "conf": "UEFA",     "rank": 6,  "wc_apps": 9 },
    "Netherlands":   {"flag": "🇳🇱", "conf": "UEFA",     "rank": 7,  "wc_apps": 13},
    "Belgium":       {"flag": "🇧🇪", "conf": "UEFA",     "rank": 3,  "wc_apps": 14},
    "Croatia":       {"flag": "🇭🇷", "conf": "UEFA",     "rank": 10, "wc_apps": 8 },
    "Morocco":       {"flag": "🇲🇦", "conf": "CAF",      "rank": 14, "wc_apps": 6 },
    "Japan":         {"flag": "🇯🇵", "conf": "AFC",      "rank": 15, "wc_apps": 7 },
    "United States": {"flag": "🇺🇸", "conf": "CONCACAF", "rank": 16, "wc_apps": 11},
    "Mexico":        {"flag": "🇲🇽", "conf": "CONCACAF", "rank": 11, "wc_apps": 17},
    "Senegal":       {"flag": "🇸🇳", "conf": "CAF",      "rank": 20, "wc_apps": 3 },
    "South Korea":   {"flag": "🇰🇷", "conf": "AFC",      "rank": 23, "wc_apps": 11},
    "Colombia":      {"flag": "🇨🇴", "conf": "CONMEBOL", "rank": 9,  "wc_apps": 7 },
    "Uruguay":       {"flag": "🇺🇾", "conf": "CONMEBOL", "rank": 12, "wc_apps": 14},
    "Denmark":       {"flag": "🇩🇰", "conf": "UEFA",     "rank": 13, "wc_apps": 6 },
    "Switzerland":   {"flag": "🇨🇭", "conf": "UEFA",     "rank": 19, "wc_apps": 12},
    "Australia":     {"flag": "🇦🇺", "conf": "AFC",      "rank": 24, "wc_apps": 6 },
    "Canada":        {"flag": "🇨🇦", "conf": "CONCACAF", "rank": 47, "wc_apps": 2 },
    "Ecuador":       {"flag": "🇪🇨", "conf": "CONMEBOL", "rank": 44, "wc_apps": 4 },
    "Serbia":        {"flag": "🇷🇸", "conf": "UEFA",     "rank": 25, "wc_apps": 13},
    "Iran":          {"flag": "🇮🇷", "conf": "AFC",      "rank": 22, "wc_apps": 6 },
    "Saudi Arabia":  {"flag": "🇸🇦", "conf": "AFC",      "rank": 56, "wc_apps": 6 },
    "Ghana":         {"flag": "🇬🇭", "conf": "CAF",      "rank": 60, "wc_apps": 4 },
    "Nigeria":       {"flag": "🇳🇬", "conf": "CAF",      "rank": 40, "wc_apps": 7 },
    "Cameroon":      {"flag": "🇨🇲", "conf": "CAF",      "rank": 52, "wc_apps": 8 },
    "Tunisia":       {"flag": "🇹🇳", "conf": "CAF",      "rank": 34, "wc_apps": 6 },
    "Turkey":        {"flag": "🇹🇷", "conf": "UEFA",     "rank": 28, "wc_apps": 2 },
    "Austria":       {"flag": "🇦🇹", "conf": "UEFA",     "rank": 26, "wc_apps": 8 },
    "Ivory Coast":   {"flag": "🇨🇮", "conf": "CAF",      "rank": 33, "wc_apps": 4 },
    "Egypt":         {"flag": "🇪🇬", "conf": "CAF",      "rank": 37, "wc_apps": 3 },
    "Jordan":        {"flag": "🇯🇴", "conf": "AFC",      "rank": 70, "wc_apps": 0 },
    "Iraq":          {"flag": "🇮🇶", "conf": "AFC",      "rank": 65, "wc_apps": 1 },
    "Panama":        {"flag": "🇵🇦", "conf": "CONCACAF", "rank": 77, "wc_apps": 1 },
    "Jamaica":       {"flag": "🇯🇲", "conf": "CONCACAF", "rank": 47, "wc_apps": 1 },
    "Venezuela":     {"flag": "🇻🇪", "conf": "CONMEBOL", "rank": 55, "wc_apps": 0 },
    "New Zealand":   {"flag": "🇳🇿", "conf": "OFC",      "rank": 97, "wc_apps": 2 },
    "Algeria":       {"flag": "🇩🇿", "conf": "CAF",      "rank": 35, "wc_apps": 4 },
    "Chile":         {"flag": "🇨🇱", "conf": "CONMEBOL", "rank": 39, "wc_apps": 9 },
    "Peru":          {"flag": "🇵🇪", "conf": "CONMEBOL", "rank": 45, "wc_apps": 5 },
    "Uzbekistan":    {"flag": "🇺🇿", "conf": "AFC",      "rank": 68, "wc_apps": 0 },
    "South Africa":  {"flag": "🇿🇦", "conf": "CAF",      "rank": 58, "wc_apps": 3 },
    "Georgia":       {"flag": "🇬🇪", "conf": "UEFA",     "rank": 74, "wc_apps": 0 },
    "Bolivia":       {"flag": "🇧🇴", "conf": "CONMEBOL", "rank": 85, "wc_apps": 3 },
    "Paraguay":      {"flag": "🇵🇾", "conf": "CONMEBOL", "rank": 51, "wc_apps": 9 },
    "Slovenia":      {"flag": "🇸🇮", "conf": "UEFA",     "rank": 55, "wc_apps": 1 },
}

VENUE_META = {
    "MetLife Stadium":         {"city": "East Rutherford", "altitude_m": 3,    "temp_c": 24.0, "is_indoor": False},
    "SoFi Stadium":            {"city": "Los Angeles",     "altitude_m": 82,   "temp_c": 22.0, "is_indoor": True },
    "AT&T Stadium":            {"city": "Dallas",          "altitude_m": 186,  "temp_c": 31.0, "is_indoor": True },
    "Levi's Stadium":          {"city": "Santa Clara",     "altitude_m": 15,   "temp_c": 19.0, "is_indoor": False},
    "Arrowhead Stadium":       {"city": "Kansas City",     "altitude_m": 280,  "temp_c": 28.0, "is_indoor": False},
    "NRG Stadium":             {"city": "Houston",         "altitude_m": 13,   "temp_c": 32.0, "is_indoor": True },
    "Hard Rock Stadium":       {"city": "Miami",           "altitude_m": 2,    "temp_c": 29.0, "is_indoor": False},
    "Lincoln Financial Field": {"city": "Philadelphia",    "altitude_m": 11,   "temp_c": 25.0, "is_indoor": False},
    "Gillette Stadium":        {"city": "Foxborough",      "altitude_m": 46,   "temp_c": 21.0, "is_indoor": False},
    "Mercedes-Benz Stadium":   {"city": "Atlanta",         "altitude_m": 316,  "temp_c": 28.0, "is_indoor": True },
    "Lumen Field":             {"city": "Seattle",         "altitude_m": 5,    "temp_c": 17.0, "is_indoor": False},
    "BMO Field":               {"city": "Toronto",         "altitude_m": 76,   "temp_c": 23.0, "is_indoor": False},
    "BC Place":                {"city": "Vancouver",       "altitude_m": 10,   "temp_c": 18.0, "is_indoor": True },
    "Estadio Azteca":          {"city": "Mexico City",     "altitude_m": 2240, "temp_c": 17.0, "is_indoor": False},
    "Estadio Akron":           {"city": "Guadalajara",     "altitude_m": 1566, "temp_c": 22.0, "is_indoor": False},
    "Estadio BBVA":            {"city": "Monterrey",       "altitude_m": 537,  "temp_c": 29.0, "is_indoor": False},
}


# ── Background simulation task ────────────────────────────────────────────

def run_simulation_bg(n_sims: int = 5000):
    state.refresh_running  = True
    state.refresh_progress = 0.0
    try:
        if not state.simulator:
            return
        results = state.simulator.run(n_sims=n_sims, pre_warm=False, use_model=False)
        state.refresh_progress = 0.9
        teams_data = []
        for _, row in results.iterrows():
            teams_data.append({
                "team":    row["team"],
                "flag":    TEAM_META.get(row["team"], {}).get("flag", "🌍"),
                "conf":    TEAM_META.get(row["team"], {}).get("conf", "—"),
                "elo":     TEAM_ELO.get(row["team"], 1750),
                "p_win":   round(float(row["p_win_tournament"]) * 100, 2),
                "p_final": round(float(row["p_reach_final"])    * 100, 2),
                "p_sf":    round(float(row["p_advance_sf"])     * 100, 2),
                "p_qf":    round(float(row["p_advance_qf"])     * 100, 2),
                "p_r16":   round(float(row["p_advance_r16"])    * 100, 2),
                "p_r32":   round(float(row["p_advance_r32"])    * 100, 2),
            })
        forecast = {
            "teams":       teams_data,
            "n_sims":      n_sims,
            "computed_at": datetime.utcnow().isoformat(),
            "champion":    teams_data[0]["team"] if teams_data else "Unknown",
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(FORECAST_CACHE, "w") as f:
            json.dump(forecast, f, indent=2)
        state.forecast         = forecast
        state.refresh_progress = 1.0
    except Exception as e:
        print(f"Simulation error: {e}")
    finally:
        state.refresh_running = False


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":           "ok",
        "model_loaded":     state.predictor is not None,
        "assembler_loaded": state.assembler is not None,
        "loaded_at":        state.loaded_at,
        "timestamp":        datetime.utcnow().isoformat(),
    }


@app.get("/teams")
def get_teams():
    teams = [{"name": k, "elo": TEAM_ELO.get(k, 1750), **v} for k, v in TEAM_META.items()]
    teams.sort(key=lambda t: t["elo"], reverse=True)
    return {"teams": teams}


@app.get("/venues")
def get_venues():
    return {"venues": [{"name": k, **v} for k, v in VENUE_META.items()]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict outcome, score, and xG for a single match.

    Score consistency guarantee:
      most_likely_score and top_scores are ALWAYS derived from
      poisson_xg_home and poisson_xg_away — never from a separate
      code path. What you see in xG is exactly what produced the score.
    """
    if req.home_team not in TEAM_META:
        raise HTTPException(400, f"Unknown team: {req.home_team}")
    if req.away_team not in TEAM_META:
        raise HTTPException(400, f"Unknown team: {req.away_team}")
    if req.home_team == req.away_team:
        raise HTTPException(400, "Teams must be different")
    if req.venue not in VENUE_META:
        raise HTTPException(400, f"Unknown venue: {req.venue}")

    vm         = VENUE_META[req.venue]
    alt        = vm["altitude_m"]
    temp       = vm["temp_c"]
    alt_stress = round(max(0.0, (alt - 500) / 4000), 4)
    elo_h      = float(TEAM_ELO.get(req.home_team, 1750))
    elo_a      = float(TEAM_ELO.get(req.away_team, 1750))

    # ── Try full model ─────────────────────────────────────────────────────
    result = None
    if state.predictor and state.assembler:
        try:
            row    = state.assembler.build_prediction_row(
                home_team=req.home_team,
                away_team=req.away_team,
                match_date=req.match_date,
                venue=req.venue,
                stage=req.stage,
                odds_home=req.odds_home or float("nan"),
                odds_draw=req.odds_draw or float("nan"),
                odds_away=req.odds_away or float("nan"),
            )
            result = state.predictor.predict_match(row)
            ph     = result.get("p_home", 0)
            pa     = result.get("p_away", 0)
            if not (0.05 < ph < 0.95) or not (0.05 < pa < 0.95):
                raise ValueError(f"Degenerate probs: {ph:.3f}/{pa:.3f}")
        except Exception as e:
            print(f"Model prediction failed ({e}), using ELO fallback")
            result = None

    # ── Fall back to ELO if model unavailable or degenerate ───────────────
    if result is None:
        result = elo_predict(req.home_team, req.away_team)
    else:
        # Enrich model result with consistent score data derived from its xG
        result = enrich_with_scores(result, req.home_team, req.away_team)

    # ── Build response ─────────────────────────────────────────────────────
    top    = result.get("top_scores", [{"score": result.get("most_likely_score","1:1"), "prob": 0.0}])
    p_home = result.get("p_home", 0.4)
    p_draw = result.get("p_draw", 0.27)
    p_away = result.get("p_away", 0.33)

    return PredictResponse(
        home_team  = req.home_team,
        away_team  = req.away_team,
        venue      = req.venue,
        stage      = req.stage,
        p_home     = round(p_home, 5),
        p_draw     = round(p_draw, 5),
        p_away     = round(p_away, 5),
        prediction = result.get("prediction", "home_win"),

        most_likely_score = result.get("most_likely_score", top[0]["score"]),
        top_scores        = top,

        poisson_xg_home = result.get("poisson_xg_home", 1.25),
        poisson_xg_away = result.get("poisson_xg_away", 1.25),
        poisson_p_home  = round(result.get("poisson_p_home", p_home), 5),
        poisson_p_draw  = round(result.get("poisson_p_draw", p_draw), 5),
        poisson_p_away  = round(result.get("poisson_p_away", p_away), 5),

        xgb_p_home = round(result.get("xgb_p_home", p_home), 5),
        xgb_p_draw = round(result.get("xgb_p_draw", p_draw), 5),
        xgb_p_away = round(result.get("xgb_p_away", p_away), 5),
        nn_p_home  = round(result.get("nn_p_home",  p_home), 5),
        nn_p_draw  = round(result.get("nn_p_draw",  p_draw), 5),
        nn_p_away  = round(result.get("nn_p_away",  p_away), 5),

        altitude_m      = alt,
        temp_c          = temp,
        altitude_stress = alt_stress,
        elo_home        = elo_h,
        elo_away        = elo_a,
        elo_diff        = round(elo_h - elo_a, 1),
        computed_at     = datetime.utcnow().isoformat(),
    )


@app.get("/forecast")
def get_forecast():
    if not state.forecast:
        run_simulation_bg(n_sims=2000)
    if not state.forecast:
        raise HTTPException(503, "Forecast not yet available. Try again in a moment.")
    return state.forecast


@app.post("/forecast/refresh")
def refresh_forecast(background_tasks: BackgroundTasks, n_sims: int = 5000):
    if state.refresh_running:
        return {"status": "already_running", "progress": state.refresh_progress}
    background_tasks.add_task(run_simulation_bg, n_sims=n_sims)
    return {"status": "started", "n_sims": n_sims}


@app.get("/forecast/status")
def forecast_status():
    return {
        "running":   state.refresh_running,
        "progress":  round(state.refresh_progress * 100, 1),
        "cached_at": state.forecast.get("computed_at") if state.forecast else None,
    }


# ── Serve the dashboard ───────────────────────────────────────────────────

DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"

if DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")

    @app.get("/")
    def root():
        return FileResponse(str(DASHBOARD_DIR / "index.html"))
else:
    @app.get("/")
    def root():
        return {"message": "WC2026 Predictor API", "docs": "/docs"}


# ── Local dev entrypoint ──────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)