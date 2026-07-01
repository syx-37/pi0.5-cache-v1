"""Compatibility check for the transformers version used with openpi patches.

V1a (2026-05-19) verified compatibility range: ``transformers>=4.53,<4.54``.

Breakage points found by empirical testing on Pi0 forward (random init,
bf16, prefix + suffix paths):

  - 4.52 and below: ``transformers.masking_utils`` module does not exist
    (``create_causal_mask`` was added in 4.53). The patched modeling_gemma.py
    raises ``ModuleNotFoundError`` at import.
  - 4.54.0+: ``LossKwargs`` was moved out of ``transformers.utils``.
    The patched modeling_gemma.py raises ``ImportError`` at import.

Within the [4.53.0, 4.53.3] range, all Pi0 forward outputs are bit-exact
identical (verified by parity_audit/v1a_smoke_test.py).

To extend the range, you must add a compatibility shim for the moved/missing
internal API (e.g. provide a local ``LossKwargs`` shim for >=4.54). Until then
this stays narrow.
"""
import warnings

import transformers

# Inclusive lower bound, exclusive upper bound.
_VERIFIED_RANGE = ("4.53", "4.54")
_VERIFIED_RANGE_STR = "transformers>=4.53,<4.54"


def _parse_version(v: str) -> tuple[int, int, int]:
    parts = v.split(".")
    major = int(parts[0]) if len(parts) > 0 else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    patch_str = parts[2] if len(parts) > 2 else "0"
    # Trim post-release suffix like "0.dev0"
    patch = int(patch_str.split(".")[0].split("rc")[0].split("dev")[0] or "0")
    return major, minor, patch


def _version_in_range(current: str) -> bool:
    cur = _parse_version(current)
    lo = _parse_version(_VERIFIED_RANGE[0])
    hi = _parse_version(_VERIFIED_RANGE[1])
    return lo[:2] <= cur[:2] < hi[:2]


def check_whether_transformers_replace_is_installed_correctly() -> bool:
    """Soft check: warns if transformers version is outside the verified range,
    but does NOT block import (returns True even when outside, on the assumption
    that the user knows what they're doing).

    Use ``EMBODIED_SKIP_TRANSFORMERS_CHECK=1`` to silence the warning entirely.
    """
    import os
    if os.environ.get("EMBODIED_SKIP_TRANSFORMERS_CHECK"):
        return True

    current = transformers.__version__
    if not _version_in_range(current):
        warnings.warn(
            f"openpi vendor patches verified for {_VERIFIED_RANGE_STR}; "
            f"current transformers={current} is outside this range. "
            f"Patches may fail to import or produce incorrect results. "
            f"See transformers_replace/models/siglip/check.py for breakage points.",
            RuntimeWarning,
            stacklevel=2,
        )
    return True
