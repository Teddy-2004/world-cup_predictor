"""
WC2026 Predictor — Database
Creates and manages the SQLite schema for all collected data.
"""

import sqlite3
from pathlib import Path
from config import DB_PATH


SCHEMA = """
-- ── Raw match results ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS matches (
    id              TEXT PRIMARY KEY,   -- football-data.org match ID
    competition     TEXT NOT NULL,       -- 'WC', 'EC', 'COPA', etc.
    season          TEXT NOT NULL,       -- '2022', '2024', etc.
    stage           TEXT,                -- 'GROUP_STAGE', 'QUARTER_FINALS', etc.
    match_date      TEXT NOT NULL,       -- ISO 8601
    home_team       TEXT NOT NULL,
    away_team       TEXT NOT NULL,
    home_goals      INTEGER,
    away_goals      INTEGER,
    home_goals_ht   INTEGER,            -- half-time
    away_goals_ht   INTEGER,
    penalties_home  INTEGER,            -- if went to shootout
    penalties_away  INTEGER,
    winner          TEXT,               -- 'HOME_TEAM'|'AWAY_TEAM'|'DRAW'
    venue_city      TEXT,
    status          TEXT,               -- 'FINISHED'|'SCHEDULED'|etc.
    source          TEXT DEFAULT 'football-data.org',
    fetched_at      TEXT DEFAULT (datetime('now'))
);

-- ── ELO ratings (ClubElo, daily snapshots) ─────────────────────────────────
CREATE TABLE IF NOT EXISTS elo_ratings (
    team            TEXT NOT NULL,
    rating_date     TEXT NOT NULL,      -- 'YYYY-MM-DD'
    elo             REAL NOT NULL,
    source          TEXT DEFAULT 'clubelo',
    fetched_at      TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (team, rating_date)
);

-- ── xG and advanced match stats (FBref) ────────────────────────────────────
CREATE TABLE IF NOT EXISTS match_stats (
    match_id        TEXT NOT NULL,      -- links to matches.id if available
    match_date      TEXT NOT NULL,
    home_team       TEXT NOT NULL,
    away_team       TEXT NOT NULL,
    home_xg         REAL,
    away_xg         REAL,
    home_shots      INTEGER,
    away_shots      INTEGER,
    home_shots_ot   INTEGER,
    away_shots_ot   INTEGER,
    home_possession REAL,
    away_possession REAL,
    home_passes     INTEGER,
    away_passes     INTEGER,
    home_pass_acc   REAL,
    away_pass_acc   REAL,
    home_pressures  INTEGER,            -- pressing actions
    away_pressures  INTEGER,
    home_ppda       REAL,               -- passes allowed per def. action
    away_ppda       REAL,
    competition     TEXT,
    source          TEXT DEFAULT 'fbref',
    fetched_at      TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (match_date, home_team, away_team)
);

-- ── Squad market values (Transfermarkt) ────────────────────────────────────
CREATE TABLE IF NOT EXISTS squad_values (
    team            TEXT NOT NULL,
    valuation_date  TEXT NOT NULL,      -- 'YYYY-MM' (monthly)
    total_value_eur REAL,               -- total squad value in EUR
    avg_value_eur   REAL,               -- per-player average
    squad_size      INTEGER,
    top11_value_eur REAL,               -- starter XI estimated value
    source          TEXT DEFAULT 'transfermarkt',
    fetched_at      TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (team, valuation_date)
);

-- ── Venue weather (Open-Meteo) ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS venue_weather (
    venue           TEXT NOT NULL,
    weather_date    TEXT NOT NULL,
    temp_max_c      REAL,
    temp_min_c      REAL,
    temp_mean_c     REAL,
    precipitation_mm REAL,
    humidity_pct    REAL,
    wind_speed_ms   REAL,
    source          TEXT DEFAULT 'open-meteo',
    fetched_at      TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (venue, weather_date)
);

-- ── FIFA rankings (monthly snapshots) ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS fifa_rankings (
    team            TEXT NOT NULL,
    ranking_date    TEXT NOT NULL,
    rank            INTEGER,
    points          REAL,
    source          TEXT DEFAULT 'fifa',
    fetched_at      TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (team, ranking_date)
);

-- ── Tournament brackets (WC 2026 groups) ───────────────────────────────────
CREATE TABLE IF NOT EXISTS wc2026_groups (
    group_name      TEXT NOT NULL,      -- 'A'..'L'
    team            TEXT NOT NULL,
    PRIMARY KEY (group_name, team)
);

-- ── Injury / suspension log ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS player_availability (
    team            TEXT NOT NULL,
    player_name     TEXT NOT NULL,
    status          TEXT NOT NULL,      -- 'injured'|'suspended'|'doubtful'
    out_from        TEXT,
    expected_return TEXT,
    source          TEXT,
    fetched_at      TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (team, player_name, out_from)
);

-- ── Head-to-head cache (derived, precomputed) ──────────────────────────────
CREATE TABLE IF NOT EXISTS h2h_cache (
    team_a          TEXT NOT NULL,
    team_b          TEXT NOT NULL,
    since_year      INTEGER DEFAULT 2000,
    matches_played  INTEGER,
    team_a_wins     INTEGER,
    draws           INTEGER,
    team_b_wins     INTEGER,
    team_a_goals    INTEGER,
    team_b_goals    INTEGER,
    computed_at     TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (team_a, team_b, since_year)
);

-- ── Useful indices ──────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_matches_date      ON matches(match_date);
CREATE INDEX IF NOT EXISTS idx_matches_teams     ON matches(home_team, away_team);
CREATE INDEX IF NOT EXISTS idx_elo_team          ON elo_ratings(team);
CREATE INDEX IF NOT EXISTS idx_stats_teams       ON match_stats(home_team, away_team);
"""


def init_db(path: Path = DB_PATH) -> sqlite3.Connection:
    """Create the database and all tables. Safe to call multiple times."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")   # better concurrent writes
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    conn.commit()
    print(f"Database ready: {path}")
    return conn


def get_conn(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


if __name__ == "__main__":
    init_db()