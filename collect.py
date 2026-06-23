"""
WC2026 Predictor — Main Data Collection Orchestrator

Run this script to collect ALL data from all sources.
Default collection is fast because weather uses a WC2026 climatology seed.
Live weather/API refreshes can take much longer due to rate limits.

Usage:
    python collect.py                  # full collection
    python collect.py --source weather # single source
    python collect.py --export         # re-export parquet only
"""

import argparse
import sqlite3
import sys
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_DIR,
    DB_PATH,
    PARQUET_DIR,
    WC2026_TEAMS,
    WC2026_VENUES,
    WC2026_GROUPS,
    FOOTBALL_DATA_API_KEY,
)
from database import init_db, get_conn
from collectors.match_data import FootballDataCollector, ClubEloCollector, InternationalResultsCollector
from collectors.advanced_stats import WeatherCollector, FBrefCollector, TransfermarktCollector


def reset_generated_data(include_raw_cache: bool = False):
    """
    Remove generated data products so a rebuild starts cleanly.

    By default this keeps data/raw/ caches because they speed up re-collection
    and are not used directly by the model. Pass --reset-cache to wipe those
    too.
    """
    targets = [
        DB_PATH,
        DB_PATH.with_suffix(".db-wal"),
        DB_PATH.with_suffix(".db-shm"),
        DATA_DIR / "features.parquet",
        DATA_DIR / "trained_models",
        DATA_DIR / "simulation_results",
    ]
    targets.extend(PARQUET_DIR.glob("*.parquet"))
    if include_raw_cache:
        targets.append(DATA_DIR / "raw")

    removed = []
    for path in targets:
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(path)

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)

    if removed:
        print("Removed generated data:")
        for path in removed:
            print(f"   {path}")
    else:
        print("No generated data to remove.")


def collect_international_results(
    conn: sqlite3.Connection,
    since_year: int = 2000,
    include_friendlies: bool = True,
):
    """Step 1: Broad senior international match results."""
    print("\n[1/6] Collecting senior international results...")
    c = InternationalResultsCollector(use_cache=True)
    df = c.fetch_results(since_year=since_year, include_friendlies=include_friendlies)
    n = c.save_to_db(df, conn)
    print(f"   -> {n:,} international match records saved")
    if not df.empty:
        print("   Top competitions:")
        print(df["competition_code"].value_counts().head(12).to_string())


def collect_matches(conn: sqlite3.Connection, since_year: int = 2000):
    """Optional official WC/Euro results from football-data.org."""
    print("\n[1/5] Collecting match results (football-data.org)...")
    if not FOOTBALL_DATA_API_KEY or FOOTBALL_DATA_API_KEY == "YOUR_FREE_API_KEY_HERE":
        print("   Skipped: FOOTBALL_DATA_API_KEY is not set.")
        print("   International-results CSV already covers the core match history.")
        return

    c = FootballDataCollector(use_cache=True)
    seasons = list(range(since_year, 2026))

    for code in tqdm(["WC", "EC"], desc="Competitions"):
        c.fetch_competition(code, seasons=seasons)

    n = c.save_to_db(conn)
    print(f"   → {n} match records saved")


def collect_elo(conn: sqlite3.Connection):
    """Step 2: derive national-team ELO ratings from collected matches."""
    print("\n[2/6] Building national-team ELO ratings from match history...")

    matches = pd.read_sql(
        """
        SELECT match_date, home_team, away_team, home_goals, away_goals, competition
        FROM matches
        WHERE status = 'FINISHED'
          AND home_goals IS NOT NULL
          AND away_goals IS NOT NULL
        ORDER BY match_date, id
        """,
        conn,
    )
    if matches.empty:
        print("   -> 0 ELO rows saved (no matches found)")
        return

    weights = {
        "WC": 1.90,
        "WCQ": 1.25,
        "EC": 1.55,
        "CA": 1.55,
        "AFCON": 1.45,
        "ASIAN_CUP": 1.45,
        "GOLD_CUP": 1.35,
        "OFC_NATIONS": 1.25,
        "CONFED": 1.35,
        "UNL": 1.05,
        "CNL": 1.00,
        "ECQ": 1.10,
        "AFCONQ": 1.05,
        "ASIAN_CUP_Q": 1.05,
        "FRIENDLY": 0.60,
    }

    ratings: dict[str, float] = {}
    rows = []

    for _, match in matches.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        home_rating = ratings.get(home, 1500.0)
        away_rating = ratings.get(away, 1500.0)

        home_goals = int(match["home_goals"])
        away_goals = int(match["away_goals"])
        if home_goals > away_goals:
            home_score = 1.0
        elif home_goals < away_goals:
            home_score = 0.0
        else:
            home_score = 0.5

        expected_home = 1.0 / (1.0 + 10.0 ** ((away_rating - home_rating) / 400.0))
        goal_diff = abs(home_goals - away_goals)
        goal_mult = 1.0 if goal_diff <= 1 else min(1.75, 1.0 + 0.15 * goal_diff)
        k = 28.0 * weights.get(str(match["competition"]), 1.0) * goal_mult
        change = k * (home_score - expected_home)

        ratings[home] = home_rating + change
        ratings[away] = away_rating - change
        rating_date = str(match["match_date"])[:10]
        fetched_at = pd.Timestamp.utcnow().isoformat()
        rows.append({
            "team": home,
            "rating_date": rating_date,
            "elo": ratings[home],
            "source": "derived-international-elo",
            "fetched_at": fetched_at,
        })
        rows.append({
            "team": away,
            "rating_date": rating_date,
            "elo": ratings[away],
            "source": "derived-international-elo",
            "fetched_at": fetched_at,
        })

    conn.executemany(
        """
        INSERT OR REPLACE INTO elo_ratings (team, rating_date, elo, source, fetched_at)
        VALUES (:team, :rating_date, :elo, :source, :fetched_at)
        """,
        rows,
    )
    conn.commit()
    print(f"   -> {len(rows):,} derived ELO rows saved for {len(ratings):,} teams")


def _fallback_venue_weather() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2026-06-01", "2026-07-31", freq="D")
    warm_venues = {"Hard Rock Stadium", "NRG Stadium", "AT&T Stadium", "Estadio BBVA"}
    altitude_venues = {"Estadio Azteca", "Estadio Akron", "Estadio BBVA"}

    for venue, meta in WC2026_VENUES.items():
        for date in dates:
            if venue in warm_venues:
                temp_mean = 29.0
                humidity = 72.0
            elif venue in altitude_venues:
                temp_mean = 21.0
                humidity = 55.0
            elif meta["city"] in {"Toronto", "Vancouver", "Seattle", "Foxborough"}:
                temp_mean = 19.0
                humidity = 62.0
            else:
                temp_mean = 24.0
                humidity = 60.0

            rows.append({
                "venue": venue,
                "weather_date": str(date.date()),
                "temp_max_c": temp_mean + 5.0,
                "temp_min_c": temp_mean - 5.0,
                "temp_mean_c": temp_mean,
                "precipitation_mm": 0.8,
                "humidity_pct": humidity,
                "wind_speed_ms": 12.0,
            })
    return pd.DataFrame(rows)


def collect_weather(conn: sqlite3.Connection, live: bool = False):
    """Step 3: Weather data for all 16 WC2026 venues"""
    print("\n[3/6] Collecting venue weather...")
    print(f"   Venues: {len(WC2026_VENUES)}")

    if not live:
        print("   Using fast WC2026 venue climatology seed. Use --live-weather to refresh Open-Meteo.")
        c = WeatherCollector(use_cache=True)
        n = c.save_to_db(_fallback_venue_weather(), conn)
        print(f"   -> {n:,} weather rows saved")
        return

    c = WeatherCollector(use_cache=True)

    # Historical (for training features on past WC matches played at similar venues)
    frames = []
    for venue in tqdm(WC2026_VENUES, desc="Venues (historical)"):
        try:
            df = c.fetch_venue_history(venue, start="2018-01-01")
            frames.append(df)
        except Exception as e:
            print(f"   [Warning] Weather history failed for {venue}: {e}")

    df_hist = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # Forecast for actual WC matches (June–July 2026)
    frames_fc = []
    for venue in tqdm(WC2026_VENUES, desc="Venues (forecast)"):
        try:
            df = c.fetch_venue_forecast(venue, days=16)
            frames_fc.append(df)
        except Exception as e:
            print(f"   [Warning] Weather forecast failed for {venue}: {e}")

    df_fc = pd.concat(frames_fc, ignore_index=True) if frames_fc else pd.DataFrame()

    combined = pd.concat([df_hist, df_fc], ignore_index=True)
    if combined.empty:
        print("   Open-Meteo unavailable; falling back to WC2026 venue climatology seed.")
        combined = _fallback_venue_weather()
    n = c.save_to_db(combined, conn)
    print(f"   → {n} weather rows saved")


def collect_xg(conn: sqlite3.Connection):
    """Step 4: xG and advanced stats from FBref"""
    print("\n[4/5] Collecting xG stats (FBref)...")
    print("   Scraping WC 2018, 2022 + Euro 2020, 2024 + Copa 2021, 2024")

    c = FBrefCollector(use_cache=True)
    frames = []

    try:
        for year in tqdm([2018, 2022], desc="World Cups"):
            df = c.fetch_wc_fixtures(year)
            frames.append(df)

        for year in tqdm([2020, 2024], desc="Euros"):
            df = c.fetch_euro_fixtures(year)
            frames.append(df)
    except Exception as e:
        print(f"\n   [Warning] FBref scraping failed: {e}")
        print("   FBref is protected by Cloudflare anti-bot security. Returning cached data or empty DataFrame.")

    valid_frames = [f for f in frames if not f.empty]
    if valid_frames:
        df_all = pd.concat(valid_frames, ignore_index=True)
        n = c.save_to_db(df_all, conn)
        print(f"   → {n} xG rows saved")
    else:
        print("   → 0 xG rows saved (could not fetch live stats)")


def collect_squad_values(conn: sqlite3.Connection):
    """Step 5: Squad market values from Transfermarkt"""
    print("\n[5/5] Collecting squad values (Transfermarkt)...")

    c = TransfermarktCollector(use_cache=True)
    try:
        df = c.fetch_wc2026_squads()
        n  = c.save_to_db(df, conn)
        print(f"   → {n} squad value rows saved")
    except Exception as e:
        print(f"\n   [Warning] Transfermarkt squad values collection failed: {e}")
        print("   → 0 squad value rows saved")


def seed_groups(conn: sqlite3.Connection):
    """Seed the WC2026 group draw into the database."""
    conn.execute("DELETE FROM wc2026_groups")
    rows = [(g, t) for g, teams in WC2026_GROUPS.items() for t in teams]
    conn.executemany("INSERT OR IGNORE INTO wc2026_groups VALUES (?, ?)", rows)
    conn.commit()
    print(f"   Group draw seeded: {len(rows)} team-group records")


def export_parquet(conn: sqlite3.Connection):
    """Export all tables to Parquet for fast feature engineering."""
    print("\nExporting to Parquet...")
    tables = ["matches", "elo_ratings", "match_stats", "squad_values",
              "venue_weather", "fifa_rankings", "wc2026_groups"]

    for table in tables:
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
        out = PARQUET_DIR / f"{table}.parquet"
        df.to_parquet(out, index=False)
        print(f"   {table}: {len(df):,} rows → {out.name}")

    print(f"\nParquet files saved to: {PARQUET_DIR}")


def print_summary(conn: sqlite3.Connection):
    """Print a summary of what's in the database."""
    print("\n" + "="*50)
    print("DATABASE SUMMARY")
    print("="*50)
    tables = ["matches", "elo_ratings", "match_stats", "squad_values",
              "venue_weather", "fifa_rankings"]
    for t in tables:
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t:<20} {n:>8,} rows")
        except Exception:
            pass
    print("="*50)


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WC2026 Data Collector")
    parser.add_argument("--source", choices=["international","matches","elo","weather","xg","squads","groups","all"],
                        default="all", help="Which source to collect")
    parser.add_argument("--since", type=int, default=2000,
                        help="Collect matches since this year (default: 2000)")
    parser.add_argument("--no-friendlies", action="store_true",
                        help="Exclude friendlies from the international-results source")
    parser.add_argument("--export", action="store_true",
                        help="Export to Parquet after collecting")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore disk cache and re-fetch everything")
    parser.add_argument("--live-weather", action="store_true",
                        help="Fetch live Open-Meteo history/forecast instead of fast climatology seed")
    parser.add_argument("--reset-generated", action="store_true",
                        help="Remove DB/parquet/features/models/simulation outputs before collecting")
    parser.add_argument("--reset-cache", action="store_true",
                        help="Also remove data/raw HTTP caches when resetting")
    args = parser.parse_args()

    print("WC2026 Match Predictor — Data Collection")
    print(f"Database: {DB_PATH}")

    if args.reset_generated:
        reset_generated_data(include_raw_cache=args.reset_cache)

    conn = init_db()

    source = args.source
    if source in ("international", "all"):
        collect_international_results(
            conn,
            since_year=args.since,
            include_friendlies=not args.no_friendlies,
        )
    if source in ("matches", "all"):       collect_matches(conn, since_year=args.since)
    if source in ("elo",     "all"):       collect_elo(conn)
    if source in ("weather", "all"):       collect_weather(conn, live=args.live_weather)
    if source in ("xg",      "all"):       collect_xg(conn)
    if source in ("squads",  "all"):       collect_squad_values(conn)

    if source in ("groups", "all"):
        seed_groups(conn)

    if args.export or source == "all":
        export_parquet(conn)

    print_summary(conn)

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
