from __future__ import annotations

import unittest

from fwc_predictor.cli import parse_args


class CliGuiTests(unittest.TestCase):
    def test_no_args_launches_gui(self) -> None:
        args = parse_args([])
        self.assertEqual(args.command, "gui")

    def test_simulate_subcommand_parses(self) -> None:
        args = parse_args(["simulate", "--sims", "500", "--seed", "42"])
        self.assertEqual(args.command, "simulate")
        self.assertEqual(args.sims, 500)
        self.assertEqual(args.seed, 42)

    def test_legacy_flags_still_run_simulation_mode(self) -> None:
        args = parse_args(["--sims", "250"])
        self.assertEqual(args.command, "simulate")
        self.assertEqual(args.sims, 250)

    def test_gui_module_imports(self) -> None:
        import fwc_predictor.gui as gui

        self.assertTrue(callable(gui.app))


if __name__ == "__main__":
    unittest.main()
