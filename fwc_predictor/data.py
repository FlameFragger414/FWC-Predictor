from __future__ import annotations

import csv
import datetime as dt
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ELO_BASE_URL = "https://www.eloratings.net"
THIRD_PLACE_TEMPLATE_URL = (
    "https://en.wikipedia.org/w/index.php?"
    "title=Template:2026_FIFA_World_Cup_third-place_table&action=raw"
)


@dataclass(frozen=True)
class Team:
    group: str
    slot: str
    code: str
    name: str


@dataclass(frozen=True)
class MatchResult:
    date: dt.date
    team1: str
    team2: str
    goals1: int
    goals2: int
    tournament: str
    venue_code: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def download_text(url: str, cache_path: Optional[Path] = None, refresh: bool = False) -> str:
    if cache_path and cache_path.exists() and not refresh:
        return cache_path.read_text(encoding="utf-8")

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "FWC Predictor research model; contact: local-use"
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        text = response.read().decode("utf-8", errors="replace")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
    return text


def load_groups(path: Path) -> Dict[str, Team]:
    teams: Dict[str, Team] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            team = Team(
                group=row["group"].strip(),
                slot=row["slot"].strip(),
                code=row["elo_code"].strip(),
                name=row["team"].strip(),
            )
            teams[team.code] = team
    return teams


def load_feature_overrides(path: Path) -> Dict[str, Dict[str, Optional[float]]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        result: Dict[str, Dict[str, Optional[float]]] = {}
        for row in reader:
            code = row.get("elo_code", "").strip()
            if not code:
                continue
            values: Dict[str, Optional[float]] = {}
            for key, value in row.items():
                if key == "elo_code":
                    continue
                value = (value or "").strip().replace(",", "")
                if not value:
                    values[key] = None
                    continue
                try:
                    values[key] = float(value)
                except ValueError:
                    values[key] = None
            result[code] = values
        return result


def load_elo_team_names(cache_dir: Path, refresh: bool = False) -> Dict[str, str]:
    text = download_text(
        f"{ELO_BASE_URL}/en.teams.tsv",
        cache_dir / "en.teams.tsv",
        refresh,
    )
    names: Dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            names[parts[0]] = parts[1]
    return names


def load_world_ratings(cache_dir: Path, refresh: bool = False) -> Dict[str, Dict[str, float]]:
    text = download_text(
        f"{ELO_BASE_URL}/World.tsv",
        cache_dir / "World.tsv",
        refresh,
    )
    ratings: Dict[str, Dict[str, float]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        try:
            ratings[parts[2]] = {
                "global_rank": float(parts[1]),
                "elo": float(parts[3]),
            }
        except ValueError:
            continue
    return ratings


def load_latest_results(cache_dir: Path, refresh: bool = False) -> List[MatchResult]:
    text = download_text(
        f"{ELO_BASE_URL}/latest.tsv",
        cache_dir / "latest.tsv",
        refresh,
    )
    matches: List[MatchResult] = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 9:
            continue
        try:
            match_date = dt.date(int(parts[0]), int(parts[1]), int(parts[2]))
            goals1 = int(parts[5])
            goals2 = int(parts[6])
        except ValueError:
            continue
        matches.append(
            MatchResult(
                date=match_date,
                team1=parts[3],
                team2=parts[4],
                goals1=goals1,
                goals2=goals2,
                tournament=parts[7],
                venue_code=parts[8],
            )
        )
    return matches


def world_cup_results(
    matches: Iterable[MatchResult],
    teams: Dict[str, Team],
    start_date: dt.date = dt.date(2026, 6, 11),
) -> List[MatchResult]:
    codes = set(teams)
    group_lookup = {code: team.group for code, team in teams.items()}
    wc_matches: List[MatchResult] = []
    for match in matches:
        if match.date < start_date or match.tournament != "WC":
            continue
        if match.team1 not in codes or match.team2 not in codes:
            continue
        if group_lookup[match.team1] != group_lookup[match.team2]:
            continue
        wc_matches.append(match)
    return sorted(wc_matches, key=lambda m: m.date)


def all_group_pairs(teams: Dict[str, Team]) -> Dict[str, List[Tuple[str, str]]]:
    by_group: Dict[str, List[str]] = {}
    for code, team in teams.items():
        by_group.setdefault(team.group, []).append(code)
    pairs: Dict[str, List[Tuple[str, str]]] = {}
    for group, codes in by_group.items():
        codes = sorted(codes, key=lambda c: teams[c].slot)
        group_pairs: List[Tuple[str, str]] = []
        for i, first in enumerate(codes):
            for second in codes[i + 1 :]:
                group_pairs.append((first, second))
        pairs[group] = group_pairs
    return pairs


def clean_wiki_markup(text: str) -> str:
    text = re.sub(r"'''+([A-L])'''", r"\1", text)
    text = re.sub(r"'''([A-L])'''", r"\1", text)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text


def load_third_place_assignments(
    cache_dir: Path,
    refresh: bool = False,
) -> Dict[str, Dict[str, str]]:
    """Return exact FIFA third-place assignment table.

    Keys are the sorted eight group letters whose third-placed teams advance.
    Values map a group winner A/B/D/E/G/I/K/L to the third-place group it faces.
    """
    raw = download_text(
        THIRD_PLACE_TEMPLATE_URL,
        cache_dir / "third_place_table.raw.wiki",
        refresh,
    )
    assignments: Dict[str, Dict[str, str]] = {}
    winner_columns = ["A", "B", "D", "E", "G", "I", "K", "L"]

    for chunk in raw.split("|-"):
        if "! scope=\"row\"" not in chunk:
            continue
        row_number = re.search(r"!\s*scope=\"row\"\s*\|\s*(\d+)", chunk)
        if not row_number:
            continue
        advancing = tuple(re.findall(r"'''([A-L])'''", chunk))
        assigned = tuple(re.findall(r"\b3([A-L])\b", clean_wiki_markup(chunk)))
        if len(advancing) != 8 or len(assigned) != 8:
            continue
        key = "".join(sorted(advancing))
        assignments[key] = dict(zip(winner_columns, assigned, strict=True))

    if len(assignments) != 495:
        raise ValueError(
            f"Expected 495 third-place combinations, parsed {len(assignments)}"
        )
    return assignments


def signed_float(value: str) -> Optional[float]:
    value = value.replace("\u2212", "-").replace("+", "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None
