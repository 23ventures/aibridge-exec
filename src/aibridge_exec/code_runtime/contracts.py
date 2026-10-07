from __future__ import annotations

class PolicyViolation(ValueError):
    """Deterministic sandbox/transform policy rejection.

    This exception belongs to the execution boundary, not to an orchestrator
    decision-policy implementation.
    """

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        self.code = code
