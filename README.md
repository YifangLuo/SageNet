# SageNetGW

## Overview

SageNetGW is a Python package for emulating stochastic gravitational-wave background
(SGWB) spectra and computing the corresponding effective extra radiation contribution,
`Delta N_eff`, from the predicted spectrum.

The current interface uses the `delta_N_eff` naming convention consistently. Legacy
`dnnu`-style output keys and helper names are no longer used in the public example code.

## Installation

```bash
pip install sagenetgw
```

## Minimal two-case test

The following script tests two representative inputs:

1. **Case A**: a Cobaya-style sample directly converted to physical SageNet input values.
2. **Case B**: a direct physical SageNet input example.

The script checks the returned spectrum, the sorted frequency grid, the computed
`Delta N_eff`, and the fixed gate behavior. If `Delta N_eff_raw > 5`, the sample is
marked as rejected and the returned spectrum is masked with `NaN` values.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["SAGENET_DELTA_N_EFF_VERBOSE_REJECTION"] = "1"

import numpy as np
from matplotlib import pyplot as plt

from sagenetgw.classes import GWPredictor


# =============================================================================
# Case A: original Cobaya-style sample, directly converted to physical params
# =============================================================================
params_case_A = {
    "r": 10 ** (-33.99478),
    "n_t": 4.2286621,
    "kappa10": 10 ** (-0.49245602),
    "T_re": 10 ** 5.3597445,
    "DN_re": 1.6802285,
    "Omega_bh2": 0.073688103,
    "Omega_ch2": 0.08447487,
    "H0": 33.906496,
    "A_s": np.exp(3.8834262) * 1e-10,
}


# =============================================================================
# Case B: direct physical SageNet input
# =============================================================================
params_case_B = {
    "r": 3.9585109e-05,
    "n_t": 1.0116972,
    "kappa10": 1.42477,
    "T_re": 0.17453859,
    "DN_re": 39.366618,
    "Omega_bh2": 0.0223828,
    "Omega_ch2": 0.1201075,
    "H0": 67.32117,
    "A_s": 2.100549e-9,
}


def run_one_case(case_name, predictor, params):
    print("\n" + "=" * 80)
    print(case_name)
    print("=" * 80)

    prediction = predictor.predict(params)

    f = np.asarray(prediction["f"], dtype=float)
    log10OmegaGW = np.asarray(prediction["log10OmegaGW"], dtype=float)

    delta_N_eff = float(prediction["delta_N_eff"])
    delta_N_eff_raw = float(prediction["delta_N_eff_raw"])
    delta_N_eff_g2 = float(prediction["delta_N_eff_g2"])
    rejected = bool(prediction["delta_N_eff_rejected"])
    reason = prediction["delta_N_eff_rejected_reason"]

    print("f shape:", f.shape)
    print("log10OmegaGW shape:", log10OmegaGW.shape)
    print("f sorted:", bool(np.all(np.diff(f) > 0)))
    print("spectrum_sorted:", prediction.get("spectrum_sorted", None))
    print("delta_N_eff_f_mode:", prediction.get("delta_N_eff_f_mode", None))

    print("\n=== Delta N_eff output ===")
    print("delta_N_eff      =", delta_N_eff)
    print("delta_N_eff_raw  =", delta_N_eff_raw)
    print("delta_N_eff_g2   =", delta_N_eff_g2)
    print("rejected         =", rejected)
    print("reason           =", reason)

    print("\n=== Gate check ===")
    print("raw > 5:", bool(delta_N_eff_raw > 5.0))
    print("delta_N_eff is NaN:", bool(np.isnan(delta_N_eff)))
    print("spectrum all NaN:", bool(np.all(np.isnan(log10OmegaGW))))

    if rejected:
        print("\nSpectrum rejected by Delta N_eff gate; no physical curve is plotted.")
    else:
        plt.figure(figsize=(7, 5))
        plt.plot(f, log10OmegaGW, "--", marker=".")
        plt.xlabel("f or log10(f)")
        plt.ylabel(r"$\log_{10}\Omega_{\rm GW}$")
        plt.title(
            rf"{case_name}: "
            rf"$\Delta N_{{\rm eff}}$={delta_N_eff:.6g}, "
            rf"$\Delta N_{{\rm eff,raw}}$={delta_N_eff_raw:.6g}"
        )
        plt.tight_layout()
        plt.show()

    return prediction


predictor = GWPredictor(
    model_type="Transformer",
    device="cpu",
)

out_A = run_one_case(
    case_name="Case A: original row_index=5632",
    predictor=predictor,
    params=params_case_A,
)

out_B = run_one_case(
    case_name="Case B: direct physical input",
    predictor=predictor,
    params=params_case_B,
)


print("\n" + "=" * 80)
print("Final compact summary")
print("=" * 80)

for name, out in [
    ("Case A", out_A),
    ("Case B", out_B),
]:
    f = np.asarray(out["f"], dtype=float)
    y = np.asarray(out["log10OmegaGW"], dtype=float)

    print(f"\n{name}")
    print("f sorted:", bool(np.all(np.diff(f) > 0)))
    print("delta_N_eff:", out["delta_N_eff"])
    print("delta_N_eff_raw:", out["delta_N_eff_raw"])
    print("rejected:", out["delta_N_eff_rejected"])
    print("reason:", out["delta_N_eff_rejected_reason"])
    print("spectrum all NaN:", bool(np.all(np.isnan(y))))
```

## Returned prediction keys

`GWPredictor.predict(params)` returns the original spectrum keys plus the new
`Delta N_eff` fields:

```python
prediction["f"]
prediction["log10OmegaGW"]
prediction["delta_N_eff"]
prediction["delta_N_eff_raw"]
prediction["delta_N_eff_g2"]
prediction["delta_N_eff_rejected"]
prediction["delta_N_eff_rejected_reason"]
prediction["delta_N_eff_diagnostics"]
```

The helper also marks the sorted spectrum state:

```python
prediction["spectrum_sorted"]
prediction["delta_N_eff_f_mode"]
```

## Gate behavior

The internal fixed gate is:

```python
Delta N_eff_raw > 5  -> rejected = True
Delta N_eff          -> NaN
log10OmegaGW         -> all NaN
```

This prevents extrapolated spectra from being accidentally plotted or saved as physical
SGWB predictions.

## Batch prediction

For a list of input dictionaries, use:

```python
batch = predictor.predict_batch_with_delta_N_eff([params_case_A, params_case_B])

print(batch["delta_N_eff"])
print(batch["delta_N_eff_raw"])
print(batch["delta_N_eff_rejected"])
print(batch["delta_N_eff_rejected_reason"])
```

## Naming convention

Use the new `delta_N_eff` names consistently. Do not use the old `dnnu` keys in new code.

| Old name | New name |
|---|---|
| `dnnu` | `delta_N_eff` |
| `dnnu_raw` | `delta_N_eff_raw` |
| `dnnu_g2` | `delta_N_eff_g2` |
| `dnnu_rejected` | `delta_N_eff_rejected` |
| `dnnu_rejected_reason` | `delta_N_eff_rejected_reason` |
| `dnnu_diagnostics` | `delta_N_eff_diagnostics` |
| `predict_batch_with_dnnu()` | `predict_batch_with_delta_N_eff()` |
| `SAGENET_DNNU_VERBOSE_REJECTION` | `SAGENET_DELTA_N_EFF_VERBOSE_REJECTION` |
