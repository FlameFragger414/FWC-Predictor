from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from .data import MatchResult, Team
from .exceptions import DataValidationError

EXPECTED_GROUPS = tuple("ABCDEFGHIJKL")
EXPECTED_TEAM_COUNT = 48
EXPECTED_TEAMS_PER_GROUP = 4


@dataclass(frozen=True)
class DataQualityWarning:
    severity: str
    area: str
    message: str


def validate_groups(teams: Dict[str, Team]) -> None:
    if len(teams) != EXPECTED_TEAM_COUNT:
        raise DataValidationError(
            f"Expected {EXPECTED_TEAM_COUNT} teams, found {len(teams)}."
        )
    by_group: Dict[str, List[Team]] = {}
    seen_names: set[str] = set()
    for code, team in teams.items():
        if not code or not team.name:
            raise DataValidationError("Group file contains a blank team code or name.")
        if team.group not in EXPECTED_GROUPS:
            raise DataValidationError(f"Unexpected group {team.group!r} for {team.name}.")
        if team.name in seen_names:
            raise DataValidationError(f"Duplicate team name in group file: {team.name}.")
        seen_names.add(team.name)
        by_group.setdefault(team.group, []).append(team)

    for group in EXPECTED_GROUPS:
        rows = by_group.get(group, [])
        if len(rows) != EXPECTED_TEAMS_PER_GROUP:
            raise DataValidationError(
                f"Group {group} must contain {EXPECTED_TEAMS_PER_GROUP} teams; found {len(rows)}."
            )
        expected_slots = {f"{group}{index}" for index in range(1, 5)}
        actual_slots = {team.slot for team in rows}
        if actual_slots != expected_slots:
            raise DataValidationError(
                f"Group {group} slots must be {sorted(expected_slots)}; "
                f"found {sorted(actual_slots)}."
            )


def validate_feature_file(path: Path, teams: Dict[str, Team]) -> None:
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "elo_code" not in reader.fieldnames:
            raise DataValidationError("Feature override file must include an elo_code column.")
        seen: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            code = (row.get("elo_code") or "").strip()
            if not code:
                raise DataValidationError(f"Blank elo_code in feature row {row_number}.")
            if code not in teams:
                raise DataValidationError(
                    f"Feature row {row_number} uses unknown team code {code}."
                )
            if code in seen:
                raise DataValidationError(f"Duplicate feature override for team code {code}.")
            seen.add(code)


def validate_completed_matches(
    matches: Iterable[MatchResult],
    teams: Dict[str, Team],
) -> List[MatchResult]:
    group_lookup = {code: team.group for code, team in teams.items()}
    seen_pairs: set[tuple[str, str]] = set()
    validated: List[MatchResult] = []
    for match in matches:
        if match.team1 not in teams or match.team2 not in teams:
            raise DataValidationError(
                f"Completed match references unknown team: {match.team1} vs {match.team2}."
            )
        if match.team1 == match.team2:
            raise DataValidationError(f"Completed match uses the same team twice: {match.team1}.")
        if group_lookup[match.team1] != group_lookup[match.team2]:
            raise DataValidationError(
                f"Completed group match crosses groups: {match.team1} vs {match.team2}."
            )
        if match.goals1 < 0 or match.goals2 < 0:
            raise DataValidationError("Completed matches cannot contain negative goals.")
        pair = (
            min(match.team1, match.team2),
            max(match.team1, match.team2),
        )
        if pair in seen_pairs:
            raise DataValidationError(
                f"Duplicate completed group match: {match.team1} vs {match.team2}."
            )
        seen_pairs.add(pair)
        validated.append(match)
    return validated


def build_data_quality_warnings(
    teams: Dict[str, Team],
    ratings: Dict[str, Dict[str, float]],
    latest_results: Iterable[MatchResult],
    feature_overrides: Dict[str, Dict[str, float | None]],
    used_snapshot_fallback: bool,
) -> List[DataQualityWarning]:
    warnings: List[DataQualityWarning] = []
    missing_ratings = [team.name for code, team in teams.items() if code not in ratings]
    if missing_ratings:
        warnings.append(
            DataQualityWarning(
                "error",
                "ratings",
                f"Missing Elo ratings for {len(missing_ratings)} teams: "
                f"{', '.join(missing_ratings[:8])}.",
            )
        )
    if used_snapshot_fallback:
        warnings.append(
            DataQualityWarning(
                "warning",
                "data freshness",
                "Live/cached Elo inputs were unavailable, so the checked-in "
                "diagnostic snapshot was used.",
            )
        )
    if not feature_overrides:
        warnings.append(
            DataQualityWarning(
                "info",
                "squad features",
                "Optional squad, tactical, injury, and country feature overrides are empty.",
            )
        )
    latest_dates = [match.date for match in latest_results]
    if latest_dates:
        latest = max(latest_dates)
        warnings.append(
            DataQualityWarning(
                "info",
                "results cache",
                f"Latest match result in the source feed is {latest.isoformat()}.",
            )
        )
    else:
        warnings.append(
            DataQualityWarning(
                "warning",
                "results cache",
                "No recent-results feed was available; recent form uses neutral fallbacks.",
            )
        )
    return warnings
