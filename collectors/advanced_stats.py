"""
WC2026 Predictor — Environment & Advanced Stats Collectors

Sources:
  3. Open-Meteo      → historical & forecast weather per venue (free, no key)
  4. FBref           → xG, pressing, possession stats (scraped, be polite)
  5. Transfermarkt   → squad market values (scraped)
"""

import re
import sqlite3
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

from collectors.base import BaseCollector
from config import OPEN_METEO_BASE, OPEN_METEO_FORECAST, WC2026_VENUES


# ══════════════════════════════════════════════════════════════════════════════
# 3. Open-Meteo — weather data
# ══════════════════════════════════════════════════════════════════════════════

class WeatherCollector(BaseCollector):
    """
    Fetches historical and forecast weather for all WC2026 venues.

    Open-Meteo is a free, no-key-required weather API.
    Docs: https://open-meteo.com/en/docs

    Variables fetched per day:
      - temperature_2m_max / min / mean
      - precipitation_sum
      - relative_humidity_2m_mean
      - wind_speed_10m_max
    """

    source_name = "open_meteo"

    DAILY_VARS = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
        "relative_humidity_2m_max",
    ]

    def fetch_venue_history(
        self, venue_name: str, start: str = "2018-01-01", end: str = None
    ) -> pd.DataFrame:
        """
        Fetch historical daily weather for a venue.
        start / end: 'YYYY-MM-DD'
        """
        end    = end or datetime.utcnow().strftime("%Y-%m-%d")
        venue  = WC2026_VENUES[venue_name]

        params = {
            "latitude":       venue["lat"],
            "longitude":      venue["lon"],
            "start_date":     start,
            "end_date":       end,
            "daily":          ",".join(self.DAILY_VARS),
            "timezone":       "UTC",
        }

        self.log.info(f"Weather history: {venue_name} ({start} → {end})")
        data   = self.get_json(OPEN_METEO_BASE, params=params)
        daily  = data.get("daily", {})
        dates  = daily.get("time", [])
        if not dates:
            return pd.DataFrame()

        df = pd.DataFrame({
            "venue":          venue_name,
            "weather_date":   dates,
            "temp_max_c":     daily.get("temperature_2m_max"),
            "temp_min_c":     daily.get("temperature_2m_min"),
            "temp_mean_c":    daily.get("temperature_2m_mean"),
            "precipitation_mm": daily.get("precipitation_sum"),
            "wind_speed_ms":  daily.get("wind_speed_10m_max"),
            "humidity_pct":   daily.get("relative_humidity_2m_max"),
            "altitude_m":     venue["altitude_m"],
        })
        return df

    def fetch_venue_forecast(self, venue_name: str, days: int = 16) -> pd.DataFrame:
        """Fetch forecast weather (up to 16 days ahead) for a venue."""
        venue  = WC2026_VENUES[venue_name]
        params = {
            "latitude":  venue["lat"],
            "longitude": venue["lon"],
            "daily":     ",".join(self.DAILY_VARS),
            "timezone":  "UTC",
            "forecast_days": days,
        }
        self.log.info(f"Weather forecast: {venue_name}")
        data  = self.get_json(OPEN_METEO_FORECAST, params=params)
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        if not dates:
            return pd.DataFrame()

        return pd.DataFrame({
            "venue":          venue_name,
            "weather_date":   dates,
            "temp_max_c":     daily.get("temperature_2m_max"),
            "temp_min_c":     daily.get("temperature_2m_min"),
            "temp_mean_c":    daily.get("temperature_2m_mean"),
            "precipitation_mm": daily.get("precipitation_sum"),
            "wind_speed_ms":  daily.get("wind_speed_10m_max"),
            "humidity_pct":   daily.get("relative_humidity_2m_max"),
            "altitude_m":     venue["altitude_m"],
        })

    def fetch_all_venues(self, start: str = "2018-01-01") -> pd.DataFrame:
        """Fetch historical weather for every WC2026 venue."""
        frames = []
        for name in WC2026_VENUES:
            df = self.fetch_venue_history(name, start=start)
            if not df.empty:
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def save_to_db(self, df: pd.DataFrame, conn: sqlite3.Connection) -> int:
        if df.empty:
            return 0
        sql = """
            INSERT OR REPLACE INTO venue_weather
            (venue, weather_date, temp_max_c, temp_min_c, temp_mean_c,
             precipitation_mm, humidity_pct, wind_speed_ms, fetched_at)
            VALUES
            (:venue,:weather_date,:temp_max_c,:temp_min_c,:temp_mean_c,
             :precipitation_mm,:humidity_pct,:wind_speed_ms,:fetched_at)
        """
        rows = df.to_dict("records")
        for r in rows:
            r["fetched_at"] = datetime.utcnow().isoformat()
        conn.executemany(sql, rows)
        conn.commit()
        self.log.info(f"Saved {len(rows)} weather rows")
        return len(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 4. FBref — xG and advanced match stats
# ══════════════════════════════════════════════════════════════════════════════

class FBrefCollector(BaseCollector):
    """
    Scrapes xG, pressing, and possession data from FBref.com.

    FBref is stathead's public site — it does NOT have an official API.
    Rules to stay polite:
      - 4 second gap between requests (RATE_LIMITS['fbref'] = 4.0)
      - Use the disk cache heavily (use_cache=True by default)
      - Only scrape what you need

    Key URLs:
      World Cup 2022 scores+xG:
        https://fbref.com/en/comps/1/schedule/World-Cup-Scores-and-Fixtures
      International matches:
        https://fbref.com/en/comps/1/history/World-Cup-Seasons
    """

    source_name = "fbref"

    FBREF_BASE = "https://fbref.com"

    def fetch_wc_fixtures(self, year: int) -> pd.DataFrame:
        """
        Scrape the score + xG table for a World Cup year.
        Returns a DataFrame with one row per match.
        """
        # FBref URL pattern for WC
        url = f"{self.FBREF_BASE}/en/comps/1/{year}/schedule/{year}-World-Cup-Scores-and-Fixtures"
        self.log.info(f"Scraping FBref WC {year} fixtures")
        html = self.get_text(url)
        return self._parse_fixtures_table(html, competition=f"WC{year}")

    def fetch_euro_fixtures(self, year: int) -> pd.DataFrame:
        url = f"{self.FBREF_BASE}/en/comps/676/{year}/schedule/{year}-UEFA-Euro-Scores-and-Fixtures"
        self.log.info(f"Scraping FBref Euro {year}")
        html = self.get_text(url)
        return self._parse_fixtures_table(html, competition=f"EURO{year}")

    def _parse_fixtures_table(self, html: str, competition: str) -> pd.DataFrame:
        """Parse the scores/xG table from FBref HTML."""
        soup   = BeautifulSoup(html, "lxml")
        tables = pd.read_html(html, attrs={"id": re.compile(r"sched_")})

        if not tables:
            self.log.warning("No schedule table found on FBref page")
            return pd.DataFrame()

        df = tables[0].copy()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Standardise column names (FBref changes them occasionally)
        rename_map = {
            "home":   "home_team",
            "away":   "away_team",
            "date":   "match_date",
            "xg":     "home_xg",
            "xg.1":   "away_xg",
            "poss":   "home_possession",
            "poss.1": "away_possession",
            "score":  "score_raw",
            "attendance": "attendance",
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

        # Drop header repeat rows that FBref injects
        if "match_date" in df.columns:
            df = df[df["match_date"].notna() & (df["match_date"] != "Date")]

        # Parse score into goals
        if "score_raw" in df.columns:
            goals = df["score_raw"].str.extract(r"(\d+)[–\-](\d+)")
            df["home_goals"] = pd.to_numeric(goals[0], errors="coerce")
            df["away_goals"] = pd.to_numeric(goals[1], errors="coerce")

        # Numeric coercions
        for col in ["home_xg", "away_xg", "home_possession", "away_possession"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["competition"] = competition
        df["source"]      = "fbref"
        df["fetched_at"]  = datetime.utcnow().isoformat()

        # Keep only rows with a result
        if "home_goals" in df.columns:
            df = df[df["home_goals"].notna()]

        self.log.info(f"  Parsed {len(df)} matches from FBref")
        return df

    def save_to_db(self, df: pd.DataFrame, conn: sqlite3.Connection) -> int:
        if df.empty:
            return 0
        sql = """
            INSERT OR REPLACE INTO match_stats
            (match_date, home_team, away_team, home_xg, away_xg,
             home_possession, away_possession, competition, source, fetched_at)
            VALUES
            (:match_date,:home_team,:away_team,:home_xg,:away_xg,
             :home_possession,:away_possession,:competition,:source,:fetched_at)
        """
        rows = []
        for _, r in df.iterrows():
            row = {
                "match_date":       str(r.get("match_date", ""))[:10],
                "home_team":        r.get("home_team", ""),
                "away_team":        r.get("away_team", ""),
                "home_xg":          r.get("home_xg"),
                "away_xg":          r.get("away_xg"),
                "home_possession":  r.get("home_possession"),
                "away_possession":  r.get("away_possession"),
                "competition":      r.get("competition"),
                "source":           "fbref",
                "fetched_at":       r.get("fetched_at"),
            }
            if row["match_date"] and row["home_team"]:
                rows.append(row)

        conn.executemany(sql, rows)
        conn.commit()
        self.log.info(f"Saved {len(rows)} FBref stat rows")
        return len(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Transfermarkt — squad market values
# ══════════════════════════════════════════════════════════════════════════════

class TransfermarktCollector(BaseCollector):
    """
    Scrapes national team squad values from Transfermarkt.

    Note: Transfermarkt has anti-scraping measures. This collector:
      - Uses realistic browser headers
      - Has a 3-second rate limit
      - Caches responses to disk to avoid repeat fetches

    WC2026 squad values page:
      https://www.transfermarkt.com/weltmeisterschaft-2026/teilnehmer/pokalwettbewerb/WM26
    """

    source_name = "transfermarkt"
    BASE = "https://www.transfermarkt.de"

    # Transfermarkt needs stricter browser headers
    TM_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.transfermarkt.de/",
    }

    def fetch_wc2026_squads(self) -> pd.DataFrame:
        """Scrape the WC2026 participant squads and their total market values."""
        url = f"{self.BASE}/weltmeisterschaft-2026/teilnehmer/pokalwettbewerb/WM26"
        try:
            html = self._tm_get(url)
            df = self._parse_squad_values(html)
            if not df.empty:
                return df
        except Exception as e:
            self.log.warning(f"Failed to fetch WC2026 squads (expected if tournament hasn't started yet): {e}")

        self.log.info("Falling back to WC 2022 squads (WM22) for realistic baseline market values...")
        fallback_url = f"{self.BASE}/weltmeisterschaft-2022/teilnehmer/pokalwettbewerb/WM22"
        try:
            html = self._tm_get(fallback_url)
            return self._parse_squad_values(html)
        except Exception as fe:
            self.log.error(f"Fallback to WC 2022 squads also failed: {fe}")
            return pd.DataFrame()

    def fetch_national_team(self, team_slug: str, team_name: str) -> dict:
        """
        Fetch squad value for one national team.
        team_slug: Transfermarkt URL slug, e.g. 'deutschland' for Germany
        """
        url  = f"{self.BASE}/{team_slug}/startseite/verein/3262"
        html = self._tm_get(url)
        soup = BeautifulSoup(html, "lxml")

        value_tag = soup.select_one("div.dataValue")
        if value_tag:
            raw = value_tag.get_text(strip=True)
            eur = self._parse_value(raw)
        else:
            eur = None

        return {
            "team":            team_name,
            "valuation_date":  datetime.utcnow().strftime("%Y-%m"),
            "total_value_eur": eur,
            "source":          "transfermarkt",
            "fetched_at":      datetime.utcnow().isoformat(),
        }

    def _tm_get(self, url: str) -> str:
        """GET with Transfermarkt-specific headers."""
        cache_path = self._cache_path(url)
        if self.use_cache and cache_path.exists():
            return cache_path.read_text()
        self._wait()
        self.log.info(f"TM GET {url}")
        resp = self.session.get(url, headers=self.TM_HEADERS, timeout=30)
        resp.raise_for_status()
        cache_path.write_text(resp.text)
        return resp.text

    def _parse_squad_values(self, html: str) -> pd.DataFrame:
        """Parse the participants table on the WC2026 Transfermarkt page."""
        soup = BeautifulSoup(html, "lxml")
        rows = []
        for tr in soup.select("table.items tbody tr"):
            cols = tr.find_all("td")
            if len(cols) < 4:
                continue
            team_tag = tr.select_one("td.hauptlink a")
            
            # Find the total value tag. On some pages it's td.rechts.hauptlink,
            # on others it's td.rechts (the first one)
            val_tag = tr.select_one("td.rechts.hauptlink")
            if not val_tag:
                rechts_cols = tr.select("td.rechts")
                if rechts_cols:
                    val_tag = rechts_cols[0]

            if team_tag and val_tag:
                rows.append({
                    "team":            team_tag.get_text(strip=True),
                    "total_value_eur": self._parse_value(val_tag.get_text(strip=True)),
                    "valuation_date":  datetime.utcnow().strftime("%Y-%m"),
                    "source":          "transfermarkt",
                    "fetched_at":      datetime.utcnow().isoformat(),
                })
        self.log.info(f"Parsed {len(rows)} squad values from Transfermarkt")
        return pd.DataFrame(rows)

    @staticmethod
    def _parse_value(text: str) -> float | None:
        """Convert '€1.23bn', '1,62 Mrd. €', etc. to a float in EUR."""
        text = text.replace(",", ".").replace("€", "").strip()
        text_lower = text.lower()
        if "bn" in text_lower or "mrd" in text_lower:
            cleaned = text_lower.replace("mrd.", "").replace("mrd", "").replace("bn", "").strip()
            return float(cleaned) * 1_000_000_000
        if "mio" in text_lower or "m" in text_lower:
            cleaned = text_lower.replace("mio.", "").replace("mio", "").replace("m", "").strip()
            return float(cleaned) * 1_000_000
        if "k" in text_lower or "tsd" in text_lower:
            cleaned = text_lower.replace("tsd.", "").replace("tsd", "").replace("k", "").strip()
            return float(cleaned) * 1_000
        try:
            return float(text)
        except ValueError:
            return None

    def save_to_db(self, df: pd.DataFrame, conn: sqlite3.Connection) -> int:
        if df.empty:
            return 0
        sql = """
            INSERT OR REPLACE INTO squad_values
            (team, valuation_date, total_value_eur, source, fetched_at)
            VALUES (:team, :valuation_date, :total_value_eur, :source, :fetched_at)
        """
        rows = df.to_dict("records")
        conn.executemany(sql, rows)
        conn.commit()
        self.log.info(f"Saved {len(rows)} squad value rows")
        return len(rows)