from __future__ import annotations


class FWCPredictorError(Exception):
    """Base exception for user-facing predictor failures."""


class DataValidationError(FWCPredictorError):
    """Raised when local or remote tournament data is malformed."""


class SimulationInputError(FWCPredictorError):
    """Raised when simulation inputs violate tournament invariants."""
