"""Cancellation channel beside signed Systemic Tau.

The published motor (systemictau 4.6.1) returns the signed mean. This
package keeps that mean and adds the residual of the 2026-09-30 note:
D = |mean change| + C.
"""

from .channel import (
    NOTE_DOI,
    ChannelResult,
    cancellation_channel,
    format_report,
    minimum_calibration_windows,
    signed_tau_series,
)
from .decomposition import decompose
from .tau import tau_b_matrix

__version__ = "0.1.0"
__all__ = [
    "NOTE_DOI",
    "ChannelResult",
    "cancellation_channel",
    "decompose",
    "format_report",
    "minimum_calibration_windows",
    "signed_tau_series",
    "tau_b_matrix",
    "__version__",
]
