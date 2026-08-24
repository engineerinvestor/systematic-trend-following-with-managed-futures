"""Cryptocurrency trend-following extension for tf-trend.

Implements methodologies described in public academic and practitioner
research. It does not reproduce, and does not claim to reproduce, any
proprietary strategy of any manager or product.
"""

from .horizons import REPLICATION_HORIZONS, TSMOM_HORIZONS, resolve_horizon, resolve_horizons
from .presets import PRESETS, Preset, apply_direction, build_signals, get_preset

__all__ = [
    "PRESETS",
    "Preset",
    "REPLICATION_HORIZONS",
    "TSMOM_HORIZONS",
    "apply_direction",
    "build_signals",
    "get_preset",
    "resolve_horizon",
    "resolve_horizons",
]
