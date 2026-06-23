"""
WC2026 Predictor — Match Results & ELO Collectors

Sources:
  1. football-data.org  →  historical match results for all major international
                           competitions (free tier, requires free API key)
  2. ClubElo.com        →  daily ELO ratings for national teams since ~1872
                           (completely free, no key)
"""

import io
import hashlib
import sqlite3
from datetime import datetime, timedelta

import pandas as pd

from collectors.base import BaseCollector
from config import (
    FOOTBALL_DATA_BASE,
    FOOTBALL_DATA_API_KEY,
    COMPETITIONS,
    CLUBELO_BASE,
    INTERNATIONAL_RESULTS_CSV,
)


TOURNAMENT_CODES = {
    "FIFA World Cup": "WC",
    "FIFA World Cup qualification": "WCQ",
    "UEFA Euro": "EC",
    "UEFA Euro qualification": "ECQ",
    "Copa América": "CA",
    "Copa America": "CA",
    "Copa América qualification": "CAQ",
    "African Cup of Nations": "AFCON",
    "African Cup of Nations qualification": "AFCONQ",
    "AFC Asian Cup": "ASIAN_CUP",
    "AFC Asian Cup qualification": "ASIAN_CUP_Q",
    "CONCACAF Gold Cup": "GOLD_CUP",
    "CONCACAF Nations League": "CNL",
    "UEFA Nations League": "UNL",
    "Oceania Nations Cup": "OFC_NATIONS",
    "Confederations Cup": "CONFED",
    "Friendly": "FRIENDLY",
}


IMPORTANT_TOURNAMENTS = {
    "WC",
    "WCQ",
    "EC",
    "ECQ",
    "CA",
    "CAQ",
    "AFCON",
    "AFCONQ",
    "ASIAN_CUP",
    "ASIAN_CUP_Q",
    "GOLD_CUP",
    "CNL",
    "UNL",
    "OFC_NATIONS",
    "CONFED",
    "FRIENDLY",
}


def _standardize_team_name(team: str) -> str:
    aliases = {
        "USA": "United States",
        "United States of America": "United States",
        "Korea Republic": "South Korea",
        "Korea, South": "South Korea",
        "Iran": "Iran",
        "IR Iran": "Iran",
        "Côte d'Ivoire": "Ivory Coast",
        "Cote d'Ivoire": "Ivory Coast",
        "Curaçao": "Curacao",
        "Türkiye": "Turkey",
        "Czech Republic": "Czechia",
        "Republic of Ireland": "Ireland",
        "DR Congo": "DR Congo",
        "Democratic Republic of the Congo": "DR Congo",
        "Cape Verde": "Cape Verde",
        "Cabo Verde": "Cape Verde",
    }
    team = str(team).strip()
    return aliases.get(team, team)


def _competition_code(tournament: str) -> str:
    tournament = str(tournament).strip()
    return TOURNAMENT_CODES.get(tournament, tournament.upper().replace(" ", "_")[:30])


# ══════════════════════════════════════════════════════════════════════════════
# 1. football-data.org — match results
# ══════════════════════════════════════════════════════════════════════════════

class FootballDataCollector(BaseCollector):
    """
    Fetches match results from football-data.org.

    Free tier limits:
      - 10 requests / minute
      - Access to select competitions only (WC, Euro, Copa, etc.)
      - Data goes back to ~2000 for most competitions

    Usage:
        c = FootballDataCollector()
        c.fetch_competition("WC", seasons=[2018, 2022])
        c.save_to_db(conn)
    """

    source_name = "football_data"

    def __init__(self, api_key: str = FOOTBALL_DATA_API_KEY, **kwargs):
        super().__init__(**kwargs)
        self._api_headers = {
            "X-Auth-Token": api_key,
            **self.session.headers,
        }
        self._matches: list[dict] = []

    def _auth_get(self, endpoint: str, params: dict = None) -> dict:
        url = f"{FOOTBALL_DATA_BASE}/{endpoint}"
        return self.get_json(url, params=params, headers=self._api_headers)

    # ── Public API ─────────────────────────────────────────────────────────

    def fetch_competition(self, code: str, seasons: list[int] = None) -> list[dict]:
        """
        Fetch all finished matches for a competition.
        code    : one of the keys in config.COMPETITIONS  ('WC', 'EC', ...)
        seasons : list of years, e.g. [2018, 2022]. None = current season only.
        """
        comp_id = COMPETITIONS[code]
        seasons = seasons or [datetime.utcnow().year]
        raw = []

        for season in seasons:
            self.log.info(f"Fetching {code} {season}")
            try:
                data = self._auth_get(
                    f"competitions/{comp_id}/matches",
                    params={"season": season, "status": "FINISHED"},
                )
                matches = data.get("matches", [])
                self.log.info(f"  → {len(matches)} matches")
                for m in matches:
                    parsed = self._parse_match(m, code)
                    if parsed:
                        raw.append(parsed)
            except Exception as e:
                self.log.warning(f"Failed {code} {season}: {e}")

        self._matches.extend(raw)
        return raw

    def fetch_all_competitions(self, since_year: int = 2000):
        """Fetch all configured competitions from since_year to now."""
        current = datetime.utcnow().year
        seasons = list(range(since_year, current + 1))
        for code in COMPETITIONS:
            self.fetch_competition(code, seasons=seasons)

    def save_to_db(self, conn: sqlite3.Connection) -> int:
        """Upsert collected matches into the database. Returns rows written."""
        if not self._matches:
            self.log.warning("No matches to save.")
            return 0

        sql = """
            INSERT OR REPLACE INTO matches
            (id, competition, season, stage, match_date, home_team, away_team,
             home_goals, away_goals, home_goals_ht, away_goals_ht,
             penalties_home, penalties_away, winner, venue_city, status, fetched_at)
            VALUES
            (:id,:competition,:season,:stage,:match_date,:home_team,:away_team,
             :home_goals,:away_goals,:home_goals_ht,:away_goals_ht,
             :penalties_home,:penalties_away,:winner,:venue_city,:status,:fetched_at)
        """
        conn.executemany(sql, self._matches)
        conn.commit()
        self.log.info(f"Saved {len(self._matches)} matches to DB")
        return len(self._matches)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._matches)

    # ── Parsing ────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_match(m: dict, competition: str) -> dict | None:
        score = m.get("score", {})
        ft    = score.get("fullTime", {})
        ht    = score.get("halfTime", {})
        pen   = score.get("penalties", {})

        home_goals = ft.get("home")
        away_goals = ft.get("away")
        if home_goals is None:          # unfinished / walkover
            return None

        return {
            "id":             str(m["id"]),
            "competition":    competition,
            "season":         str(m.get("season", {}).get("startDate", "")[:4]),
            "stage":          m.get("stage"),
            "match_date":     m.get("utcDate", "")[:10],
            "home_team":      m["homeTeam"]["name"],
            "away_team":      m["awayTeam"]["name"],
            "home_goals":     home_goals,
            "away_goals":     away_goals,
            "home_goals_ht":  ht.get("home"),
            "away_goals_ht":  ht.get("away"),
            "penalties_home": pen.get("home"),
            "penalties_away": pen.get("away"),
            "winner":         score.get("winner"),
            "venue_city":     m.get("venue"),
            "status":         m.get("status"),
            "fetched_at":     datetime.utcnow().isoformat(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 2. ClubElo — ELO ratings
# ══════════════════════════════════════════════════════════════════════════════

class ClubEloCollector(BaseCollector):
    """
    Fetches ELO ratings from http://api.clubelo.com

    Endpoints (all return CSV):
      /TEAMNAME          → full history for one team
      /YYYY-MM-DD        → all teams' ratings on a given date

    National team names use their English names with no spaces, e.g.:
        Germany, France, Brazil, SouthKorea, CostaRica
    """

    source_name = "clubelo"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ratings: list[dict] = []

    # ── Public API ─────────────────────────────────────────────────────────

    def fetch_team_history(self, team: str) -> pd.DataFrame:
        """
        Download the full ELO history for one national team.
        team: ClubElo name, e.g. 'Germany', 'SouthKorea', 'CostaRica'
        """
        team_slug = team.replace(" ", "")
        url       = f"{CLUBELO_BASE}/{team_slug}"
        try:
            text = self.get_text(url)
            df   = pd.read_csv(io.StringIO(text), parse_dates=["From", "To"])
            df["team"] = team
            self.log.info(f"  {team}: {len(df)} ELO entries")
            return df
        except Exception as e:
            self.log.warning(f"Failed {team}: {e}")
            return pd.DataFrame()

    def fetch_all_wc_teams(self, teams: list[str]) -> pd.DataFrame:
        """Fetch ELO history for every team in the list."""
        frames = []
        for team in teams:
            df = self.fetch_team_history(team)
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def fetch_snapshot(self, date: str) -> pd.DataFrame:
        """
        Get ELO ratings for ALL teams on a specific date.
        date: 'YYYY-MM-DD'
        """
        url  = f"{CLUBELO_BASE}/{date}"
        text = self.get_text(url)
        df   = pd.read_csv(io.StringIO(text))
        self.log.info(f"Snapshot {date}: {len(df)} teams")
        return df

    def fetch_snapshots_range(self, start: str, end: str, freq: str = "MS") -> pd.DataFrame:
        """
        Fetch monthly snapshots between start and end dates.
        start/end : 'YYYY-MM-DD'
        freq      : pandas date offset, 'MS' = month start
        """
        dates  = pd.date_range(start, end, freq=freq).strftime("%Y-%m-%d").tolist()
        frames = []
        for d in dates:
            df = self.fetch_snapshot(d)
            if not df.empty:
                df["snapshot_date"] = d
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def save_to_db(self, df: pd.DataFrame, conn: sqlite3.Connection) -> int:
        """
        Save ELO data to the elo_ratings table.
        Expects columns: team, From (date), Elo
        """
        if df.empty:
            return 0

        rows = []
        for _, row in df.iterrows():
            # ClubElo history rows have From/To/Elo columns
            date_col = "From" if "From" in df.columns else "snapshot_date"
            rows.append({
                "team":        row.get("Club", row.get("team", "")),
                "rating_date": str(row[date_col])[:10],
                "elo":         float(row.get("Elo", row.get("elo", 0))),
                "fetched_at":  datetime.utcnow().isoformat(),
            })

        sql = """
            INSERT OR REPLACE INTO elo_ratings (team, rating_date, elo, fetched_at)
            VALUES (:team, :rating_date, :elo, :fetched_at)
        """
        conn.executemany(sql, rows)
        conn.commit()
        self.log.info(f"Saved {len(rows)} ELO rows to DB")
        return len(rows)

    def get_elo_on_date(self, conn: sqlite3.Connection, team: str, date: str) -> float | None:
        """
        Convenience: get the closest ELO rating for a team before a match date.
        Returns None if no data.
        """
        cur = conn.execute(
            """
            SELECT elo FROM elo_ratings
            WHERE team = ? AND rating_date <= ?
            ORDER BY rating_date DESC LIMIT 1
            """,
            (team, date),
        )
        row = cur.fetchone()
        return float(row[0]) if row else None


# ══════════════════════════════════════════════════════════════════════════════
# 1b. Open international results CSV
# ══════════════════════════════════════════════════════════════════════════════

class InternationalResultsCollector(BaseCollector):
    """
    Fetch senior international match results from a broad public CSV.

    This is the main training data source for the World Cup model. It avoids
    the club-data pollution that happens when football-data.org competition IDs
    such as Copa Libertadores or CONCACAF Champions Cup are mistaken for
    national-team tournaments.
    """

    source_name = "international_results"

    def fetch_results(
        self,
        since_year: int = 2000,
        include_friendlies: bool = True,
    ) -> pd.DataFrame:
        text = self.get_text(INTERNATIONAL_RESULTS_CSV)
        df = pd.read_csv(io.StringIO(text), parse_dates=["date"])
        df = df[df["date"].dt.year >= since_year].copy()

        df["competition_code"] = df["tournament"].map(_competition_code)
        allowed = set(IMPORTANT_TOURNAMENTS)
        if include_friendlies:
            allowed.add("FRIENDLY")
        else:
            allowed.discard("FRIENDLY")

        # Keep all explicit major/qualifier tournaments. Smaller tournaments
        # are still useful if they involve World Cup teams, but they should not
        # dominate the model, so leave them out of the default rebuild.
        df = df[df["competition_code"].isin(allowed)].copy()

        for col in ["home_team", "away_team"]:
            df[col] = df[col].map(_standardize_team_name)

        df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
        df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
        df = df[df["home_score"].notna() & df["away_score"].notna()].copy()
        return df

    @staticmethod
    def _row_to_match(row: pd.Series) -> dict:
        home_goals = int(row["home_score"])
        away_goals = int(row["away_score"])
        if home_goals > away_goals:
            winner = "HOME_TEAM"
        elif away_goals > home_goals:
            winner = "AWAY_TEAM"
        else:
            winner = "DRAW"

        stable_id = hashlib.md5(
            f"intl|{row['date'].date()}|{row['home_team']}|{row['away_team']}|"
            f"{home_goals}|{away_goals}|{row['tournament']}".encode()
        ).hexdigest()

        return {
            "id": stable_id,
            "competition": row["competition_code"],
            "season": str(row["date"].year),
            "stage": row["tournament"],
            "match_date": str(row["date"].date()),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_goals_ht": None,
            "away_goals_ht": None,
            "penalties_home": None,
            "penalties_away": None,
            "winner": winner,
            "venue_city": row.get("city"),
            "status": "FINISHED",
            "source": "international-results",
            "fetched_at": datetime.utcnow().isoformat(),
        }

    def save_to_db(self, df: pd.DataFrame, conn: sqlite3.Connection) -> int:
        if df.empty:
            self.log.warning("No international results to save.")
            return 0

        rows = [self._row_to_match(row) for _, row in df.iterrows()]
        sql = """
            INSERT OR REPLACE INTO matches
            (id, competition, season, stage, match_date, home_team, away_team,
             home_goals, away_goals, home_goals_ht, away_goals_ht,
             penalties_home, penalties_away, winner, venue_city, status, source, fetched_at)
            VALUES
            (:id,:competition,:season,:stage,:match_date,:home_team,:away_team,
             :home_goals,:away_goals,:home_goals_ht,:away_goals_ht,
             :penalties_home,:penalties_away,:winner,:venue_city,:status,:source,:fetched_at)
        """
        conn.executemany(sql, rows)
        conn.commit()
        self.log.info(f"Saved {len(rows)} international matches to DB")
        return len(rows)
