from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path

from fwc_predictor.data import MatchResult, Team, load_groups, project_root
from fwc_predictor.exceptions import DataValidationError
from fwc_predictor.validation import (
    validate_completed_matches,
    validate_feature_file,
    validate_groups,
)


class ValidationTests(unittest.TestCase):
    def test_group_file_validates(self) -> None:
        teams = load_groups(project_root() / "data" / "groups_2026.csv")
        validate_groups(teams)

    def test_malformed_group_count_fails(self) -> None:
        teams = {"AA": Team("A", "A1", "AA", "Alpha")}
        with self.assertRaises(DataValidationError):
            validate_groups(teams)

    def test_duplicate_feature_override_fails(self) -> None:
        teams = load_groups(project_root() / "data" / "groups_2026.csv")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.csv"
            path.write_text("elo_code,squad_rating\nMX,1\nMX,2\n", encoding="utf-8")
            with self.assertRaises(DataValidationError):
                validate_feature_file(path, teams)

    def test_duplicate_completed_match_fails(self) -> None:
        teams = load_groups(project_root() / "data" / "groups_2026.csv")
        match = MatchResult(dt.date(2026, 6, 12), "MX", "ZA", 1, 0, "WC", "MX")
        with self.assertRaises(DataValidationError):
            validate_completed_matches([match, match], teams)

    def test_cross_group_completed_match_fails(self) -> None:
        teams = load_groups(project_root() / "data" / "groups_2026.csv")
        match = MatchResult(dt.date(2026, 6, 12), "MX", "CA", 1, 0, "WC", "MX")
        with self.assertRaises(DataValidationError):
            validate_completed_matches([match], teams)


if __name__ == "__main__":
    unittest.main()
