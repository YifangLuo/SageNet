# -*- coding: utf-8 -*-


from __future__ import annotations

from typing import Tuple

import numpy as np

from .constants import Neff0, Omega_nh2, ln10


def maybe_log10f(f) -> Tuple[np.ndarray, str]:

    f = np.asarray(f, dtype=float)
    if f.size == 0:
        return f, "empty"

    fmin = np.nanmin(f)
    fmax = np.nanmax(f)

    if not np.isfinite(fmin) or not np.isfinite(fmax):
        return f, "assume_log10f_nonfinite_seen"

    if fmin <= 0:
        return f, "assume_log10f_nonpositive_seen"

    if (-200.0 < fmin < 200.0) and (-200.0 < fmax < 200.0):
        if (fmin < 0) or (fmax < 10):
            return f, "assume_log10f_by_range"

    return np.log10(f), "converted_linear_f_to_log10"


def clean_sort_unique(
    x_in,
    y_in,
    *,
    clamp_y_nonfinite_to: float = -300.0,
) -> Tuple[np.ndarray, np.ndarray]:

    x = np.asarray(x_in, dtype=float)
    y = np.asarray(y_in, dtype=float)

    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape; got {x.shape} and {y.shape}."
        )

    mx = np.isfinite(x)
    x = x[mx]
    y = y[mx]

    y = np.where(np.isfinite(y), y, float(clamp_y_nonfinite_to))

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x, idx = np.unique(x, return_index=True)
    y = y[idx]

    if x.size < 2:
        raise ValueError("Not enough valid x points after cleaning; need >=2.")

    dx = np.diff(x)
    if not np.all(dx > 0):
        raise ValueError(
            f"x is not strictly increasing after unique/sort. min(dx)={dx.min()}"
        )

    return x, y


def simpson_atol_from_dnnu_tol(H0: float, dnnu_tol_abs: float) -> float:

    h = float(H0) / 100.0
    Omega_nu_val = Omega_nh2 / (h * h)
    g2_atol = float(dnnu_tol_abs) * (Omega_nu_val / Neff0)
    return float(g2_atol / ln10)


def g2_to_dnnu(g2: float, H0: float) -> float:
    h = float(H0) / 100.0
    Omega_nu_val = Omega_nh2 / (h * h)
    return float(Neff0 * float(g2) / Omega_nu_val)


__all__ = [
    "maybe_log10f",
    "clean_sort_unique",
    "simpson_atol_from_dnnu_tol",
    "g2_to_dnnu",
]