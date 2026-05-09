# -*- coding: utf-8 -*-
"""
Internal Delta-Neff integration API for SageNet.

Design:
    SageNet prediction {f, log10OmegaGW} + H0
        -> sort/clean frequency grid
        -> PCHIP + adaptive Simpson integration
        -> Delta N_eff

Fixed non-extrapolation rule:
    Delta N_eff > 5  => extrapolated, dnnu = NaN, print a message, continue.

No user-facing interpolation options, no user-facing extrapolation threshold,
and no option to keep extrapolated curves.
"""

from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy import integrate

from .constants import ln10
from .integrator import InterpLogOmegaPCHIP, adaptive_simpson_interpolated
from .utils import (
    clean_sort_unique,
    g2_to_dnnu,
    maybe_log10f,
    simpson_atol_from_dnnu_tol,
)


_DNNU_TOL_ABS: float = 1e-6
_SIMPSON_RTOL: float = 1e-5
_SIMPSON_MAX_DEPTH: int = 35
_SIMPSON_MAX_EVALS: int = 500_000
_CLAMP_LOG10OMEGA_NONFINITE_TO: float = -300.0

_DNNU_MIN_ALLOWED: float = 0.0
_DNNU_MAX_ALLOWED: float = 5.0


@dataclass(frozen=True)
class DnnuResult:
    dnnu: float
    dnnu_raw: float
    g2: float
    rejected: bool
    rejected_reason: str
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class DnnuBatchResult:
    dnnu: np.ndarray
    dnnu_raw: np.ndarray
    g2: np.ndarray
    rejected: np.ndarray
    rejected_reason: np.ndarray
    diagnostics: List[Dict[str, Any]]


def _extract_prediction_arrays(prediction_or_f, log10OmegaGW=None):
    if isinstance(prediction_or_f, Mapping):
        if log10OmegaGW is not None:
            raise TypeError(
                "When the first argument is a prediction dict, "
                "log10OmegaGW must not be passed."
            )
        if "f" not in prediction_or_f or "log10OmegaGW" not in prediction_or_f:
            raise KeyError("prediction must contain 'f' and 'log10OmegaGW'.")
        return prediction_or_f["f"], prediction_or_f["log10OmegaGW"]

    if log10OmegaGW is None:
        raise TypeError(
            "log10OmegaGW is required when the first argument is an array."
        )

    return prediction_or_f, log10OmegaGW


def _sort_prediction_spectrum_for_output(
    prediction: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Internal helper used by GWPredictor.

    Sort, clean, and deduplicate the returned SageNet spectrum using the
    same frequency-ordering logic as the dnnu integrator.

    This function is intentionally not exported as public API.
    """
    if "f" not in prediction or "log10OmegaGW" not in prediction:
        raise KeyError("prediction must contain 'f' and 'log10OmegaGW'.")

    f_original = np.asarray(prediction["f"], dtype=float)
    y_original = np.asarray(prediction["log10OmegaGW"], dtype=float)

    if f_original.shape != y_original.shape:
        raise ValueError(
            "f and log10OmegaGW must have the same shape for one spectrum."
        )

    x_for_sort, f_mode = maybe_log10f(f_original)

    valid_x = np.isfinite(x_for_sort)
    x_valid = x_for_sort[valid_x]
    f_valid = f_original[valid_x]
    y_valid = y_original[valid_x]

    y_valid = np.where(
        np.isfinite(y_valid),
        y_valid,
        float(_CLAMP_LOG10OMEGA_NONFINITE_TO),
    )

    order = np.argsort(x_valid)
    x_sorted = x_valid[order]
    f_sorted = f_valid[order]
    y_sorted = y_valid[order]

    _, unique_idx = np.unique(x_sorted, return_index=True)
    unique_idx = np.sort(unique_idx)

    f_sorted = f_sorted[unique_idx]
    y_sorted = y_sorted[unique_idx]

    if f_sorted.size < 2:
        raise ValueError("Not enough valid spectrum points after sorting.")

    sorted_prediction = dict(prediction)
    sorted_prediction["f"] = f_sorted.tolist()
    sorted_prediction["log10OmegaGW"] = y_sorted.tolist()
    sorted_prediction["dnnu_f_mode"] = f_mode
    sorted_prediction["spectrum_sorted"] = True

    return sorted_prediction


def _dnnu_verbose_rejection_enabled() -> bool:
    return os.environ.get(
        "SAGENET_DNNU_VERBOSE_REJECTION", "0"
    ).lower() in {"1", "true", "yes", "on"}


def _print_dnnu_rejection_message(
    *,
    dnnu_raw: float,
    sample_index: Optional[int] = None,
) -> None:
    if not _dnnu_verbose_rejection_enabled():
        return

    if sample_index is None:
        msg = (
            "[SageNet dnnu] Extrapolated prediction rejected because "
            f"Delta N_eff={dnnu_raw:.8g} > 5. "
            "The corresponding dnnu is recorded as NaN."
        )
    else:
        msg = (
            f"[SageNet dnnu] sample {sample_index}: "
            "extrapolated prediction rejected because "
            f"Delta N_eff={dnnu_raw:.8g} > 5. "
            "The corresponding dnnu is recorded as NaN."
        )

    print(msg, file=sys.stderr)


def _apply_fixed_dnnu_gate(
    *,
    dnnu_raw: float,
    diag: Dict[str, Any],
    sample_index: Optional[int] = None,
) -> Tuple[float, bool, str]:
    rejected = False
    reason = ""

    if not np.isfinite(dnnu_raw):
        rejected = True
        reason = "dnnu_nonfinite"
    elif dnnu_raw < _DNNU_MIN_ALLOWED:
        rejected = True
        reason = "dnnu_below_0"
    elif dnnu_raw > _DNNU_MAX_ALLOWED:
        rejected = True
        reason = "dnnu_above_5"

    dnnu = float("nan") if rejected else float(dnnu_raw)

    diag["dnnu_raw"] = float(dnnu_raw)
    diag["dnnu"] = float(dnnu)
    diag["dnnu_rejected"] = bool(rejected)
    diag["dnnu_rejected_reason"] = reason
    diag["dnnu_extrapolated"] = bool(reason == "dnnu_above_5")
    diag["dnnu_min_allowed"] = float(_DNNU_MIN_ALLOWED)
    diag["dnnu_max_allowed"] = float(_DNNU_MAX_ALLOWED)

    if reason == "dnnu_above_5":
        _print_dnnu_rejection_message(
            dnnu_raw=float(dnnu_raw),
            sample_index=sample_index,
        )

    return float(dnnu), bool(rejected), reason


def compute_g2(
    f_like,
    log10OmegaGW_like,
    *,
    H0: float,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute:

        g2 = ln(10) * integral Omega_GW d log10(f)

    The frequency grid is automatically converted to log10(f), cleaned,
    sorted, and deduplicated before integration.
    """
    x0, f_mode = maybe_log10f(f_like)

    x, ylog = clean_sort_unique(
        x0,
        log10OmegaGW_like,
        clamp_y_nonfinite_to=_CLAMP_LOG10OMEGA_NONFINITE_TO,
    )

    omega_native = np.power(10.0, ylog)

    if not np.all(np.isfinite(omega_native)):
        raise ValueError("Omega_GW contains NaN/Inf after 10**log10OmegaGW.")

    if np.any(omega_native < 0.0):
        raise ValueError("Omega_GW became negative after 10**log10OmegaGW.")

    g2_trapz = float(integrate.trapezoid(y=omega_native, x=x) * ln10)

    simpson_atol = simpson_atol_from_dnnu_tol(
        H0=float(H0),
        dnnu_tol_abs=_DNNU_TOL_ABS,
    )

    f_interp = InterpLogOmegaPCHIP(x, ylog)

    I_raw_simp, simp_diag, eval_points = adaptive_simpson_interpolated(
        f_interp,
        float(x[0]),
        float(x[-1]),
        rtol=_SIMPSON_RTOL,
        atol=simpson_atol,
        max_depth=_SIMPSON_MAX_DEPTH,
        max_evals=_SIMPSON_MAX_EVALS,
    )

    g2_simp = float(I_raw_simp * ln10)

    g2_final = g2_simp
    method_final = "simpson_pchip"
    fallback_used = False

    bad_simpson = (
        (not bool(simp_diag.get("simpson_converged", False)))
        or (not np.isfinite(g2_simp))
        or (g2_simp < 0.0)
    )

    if bad_simpson:
        omega_eval = np.array(
            [f_interp(float(xq)) for xq in eval_points],
            dtype=float,
        )
        g2_final = float(integrate.trapezoid(y=omega_eval, x=eval_points) * ln10)
        method_final = "trapz_refined_fallback"
        fallback_used = True

    rel_diff = (
        float((g2_final - g2_trapz) / g2_trapz)
        if g2_trapz != 0.0
        else float("nan")
    )

    diag: Dict[str, Any] = {
        "f_mode": f_mode,
        "n_input_points": int(np.asarray(f_like).size),
        "n_clean_points": int(x.size),
        "dnnu_tol_abs": float(_DNNU_TOL_ABS),
        "simpson_atol_raw_from_dnnu": float(simpson_atol),
        "g2_trapz": float(g2_trapz),
        "g2_final": float(g2_final),
        "g2_rel_diff_vs_native_trapz": rel_diff,
        "method_final": method_final,
        "fallback_used": bool(fallback_used),
    }
    diag.update(simp_diag)

    return float(g2_final), diag


def compute_dnnu(
    prediction_or_f,
    log10OmegaGW=None,
    *,
    H0: float,
    sample_index: Optional[int] = None,
) -> DnnuResult:
    """
    Compute Delta N_eff for one SageNet spectrum.

    If Delta N_eff > 5:
        - print a message
        - return dnnu = NaN
        - rejected = True
        - rejected_reason = "dnnu_above_5"
        - do not raise
    """
    f_like, log_like = _extract_prediction_arrays(
        prediction_or_f,
        log10OmegaGW=log10OmegaGW,
    )

    g2, diag = compute_g2(
        f_like,
        log_like,
        H0=float(H0),
    )

    dnnu_raw = float(g2_to_dnnu(g2, H0=float(H0)))

    dnnu, rejected, reason = _apply_fixed_dnnu_gate(
        dnnu_raw=dnnu_raw,
        diag=diag,
        sample_index=sample_index,
    )

    return DnnuResult(
        dnnu=float(dnnu),
        dnnu_raw=float(dnnu_raw),
        g2=float(g2),
        rejected=bool(rejected),
        rejected_reason=reason,
        diagnostics=diag,
    )


def _as_h0_vector(H0, n_samples: int) -> np.ndarray:
    h0 = np.asarray(H0, dtype=float)

    if h0.ndim == 0:
        return np.full(n_samples, float(h0), dtype=float)

    h0 = h0.reshape(-1)

    if h0.size != n_samples:
        raise ValueError(
            f"H0 must be scalar or have length {n_samples}; got length {h0.size}."
        )

    return h0.astype(float)


def _select_spectrum_row(
    f_arr: np.ndarray,
    y_arr: np.ndarray,
    i: int,
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if y_arr.ndim == 1:
        y_i = y_arr
    elif y_arr.ndim == 2:
        y_i = y_arr[i]
    else:
        raise ValueError(
            "log10OmegaGW must be 1D for one spectrum or 2D for batch spectra."
        )

    if f_arr.ndim == 1:
        f_i = f_arr
    elif f_arr.ndim == 2 and f_arr.shape[0] == n_samples:
        f_i = f_arr[i]
    else:
        raise ValueError(
            "f must be a 1D shared frequency grid or a 2D array with the same "
            "batch size as log10OmegaGW."
        )

    return f_i, y_i


def compute_dnnu_batch(
    prediction_or_f,
    log10OmegaGW=None,
    *,
    H0,
) -> DnnuBatchResult:
    """
    Compute Delta N_eff for a batch/matrix of spectra.

    If one sample has Delta N_eff > 5:
        dnnu[i] = NaN
        rejected[i] = True
        rejected_reason[i] = "dnnu_above_5"

    The loop continues.
    """
    f_like, log_like = _extract_prediction_arrays(
        prediction_or_f,
        log10OmegaGW=log10OmegaGW,
    )

    f_arr = np.asarray(f_like, dtype=float)
    y_arr = np.asarray(log_like, dtype=float)

    if y_arr.ndim == 1:
        n_samples = 1
    elif y_arr.ndim == 2:
        n_samples = int(y_arr.shape[0])
    else:
        raise ValueError(
            "log10OmegaGW must be 1D for one spectrum or 2D for batch spectra."
        )

    h0_vec = _as_h0_vector(H0, n_samples)

    dnnu = np.full(n_samples, np.nan, dtype=float)
    dnnu_raw = np.full(n_samples, np.nan, dtype=float)
    g2 = np.full(n_samples, np.nan, dtype=float)
    rejected = np.zeros(n_samples, dtype=bool)
    rejected_reason = np.full(n_samples, "", dtype=object)
    diagnostics: List[Dict[str, Any]] = []

    for i in range(n_samples):
        f_i, y_i = _select_spectrum_row(
            f_arr=f_arr,
            y_arr=y_arr,
            i=i,
            n_samples=n_samples,
        )

        result = compute_dnnu(
            f_i,
            y_i,
            H0=float(h0_vec[i]),
            sample_index=i if n_samples > 1 else None,
        )

        dnnu[i] = result.dnnu
        dnnu_raw[i] = result.dnnu_raw
        g2[i] = result.g2
        rejected[i] = result.rejected
        rejected_reason[i] = result.rejected_reason
        diagnostics.append(result.diagnostics)

    return DnnuBatchResult(
        dnnu=dnnu,
        dnnu_raw=dnnu_raw,
        g2=g2,
        rejected=rejected,
        rejected_reason=rejected_reason,
        diagnostics=diagnostics,
    )


def compute_dnnu_from_predictor(
    predictor,
    params: Mapping[str, float],
) -> DnnuResult:
    if "H0" not in params:
        raise KeyError("params must contain 'H0'.")

    prediction = predictor.predict(dict(params))

    return compute_dnnu(
        prediction,
        H0=float(params["H0"]),
    )


__all__ = [
    "DnnuResult",
    "DnnuBatchResult",
    "compute_g2",
    "compute_dnnu",
    "compute_dnnu_batch",
    "compute_dnnu_from_predictor",
]