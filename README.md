# SageNetGW

## Overview

SageNetGW, also referred to as SageNet+, is a Python package for fast emulation of the inflationary stochastic gravitational wave background (SGWB) spectrum with stiff-era amplification. It extends the SageNet framework described in Zhang et al. (2025), combining neural-network emulators with the numerical `stiffGWpy` backend.

The package supports neural-network prediction of SGWB spectra using models such as Transformer, LSTM, RNN, GRU, CNN, TCN, and CosmicNet2. It also keeps a numerical backend through `stiffGWpy` for direct SGWB calculations.

The latest version also provides an integrated `predict_with_dnnu()` interface, which returns a sorted SGWB spectrum and computes the corresponding contribution to the effective number of relativistic species, `Delta N_eff`.

Main features:

- Fast neural-network prediction of SGWB spectra.
- Optional numerical SGWB calculation through `stiffGWpy`.
- Automatic spectrum sorting and cleaning in the new `predict_with_dnnu()` interface.
- Built-in `Delta N_eff` integration from the predicted SGWB spectrum.
- Automatic rejection of extrapolative spectra with `Delta N_eff > 5`.
- User-facing `dnnu = NaN` and masked `log10OmegaGW` for rejected spectra.
- Backward-compatible `predict()` interface.

For the original SageNet and stiffGWpy projects, see:

- [SageNet](https://github.com/YifangLuo/SageNet)
- [stiffGWpy](https://github.com/bohuarolandli/stiffGWpy)

---

## Installation

### Install from PyPI

```bash
pip install sagenetgw
```

### Install the latest source version

If you want the latest development version with built-in `Delta N_eff` support, clone the repository with submodules:

```bash
git clone --recurse-submodules https://github.com/Hdiao112/SageNet.git
cd SageNet
python -m pip install -e .
```

If the `stiffGWpy` submodule was not downloaded, run:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

You should see files such as:

```text
sagenetgw/stiffGWpy/global_param.py
sagenetgw/stiffGWpy/th.dat
sagenetgw/stiffGWpy/stiff_SGWB.py
```

---

## Dependencies

- Python >= 3.8
- PyTorch
- NumPy
- SciPy
- scikit-learn
- Astropy
- Matplotlib, for plotting examples

A typical installation is:

```bash
python -m pip install numpy scipy scikit-learn astropy torch matplotlib
python -m pip install -e .
```

---

## Quick Start: SGWB Spectrum Prediction

The original `predict()` interface is preserved. It returns the raw SageNet prediction:

```python
from sagenetgw.classes import GWPredictor
import numpy as np
from matplotlib import pyplot as plt

predictor = GWPredictor(
    model_type="Transformer",
    device="cpu",
)

prediction = predictor.predict({
    "r":         3.9585109e-05,
    "n_t":       1.0116972,
    "kappa10":   110.42477,
    "T_re":      0.17453859,
    "DN_re":     39.366618,
    "Omega_bh2": 0.0223828,
    "Omega_ch2": 0.1201075,
    "H0":        67.32117,
    "A_s":       2.100549e-9,
})

pred_coords = np.column_stack(
    (prediction["f"], prediction["log10OmegaGW"])
)

plt.plot(
    pred_coords[:, 0],
    pred_coords[:, 1],
    "--",
    color="royalblue",
    marker=".",
)
plt.xlabel("f or log10(f)")
plt.ylabel(r"$\log_{10}\Omega_{\rm GW}$")
plt.tight_layout()
plt.show()
```

This interface is kept for backward compatibility. The returned frequency grid may follow the native model output order.

---

## New Interface: Prediction with Built-in `Delta N_eff`

The recommended interface for new applications is:

```python
prediction = predictor.predict_with_dnnu(params)
```

This interface automatically:

1. Predicts the SGWB spectrum.
2. Sorts and cleans the returned frequency grid.
3. Computes `Delta N_eff` from the predicted spectrum.
4. Rejects spectra with `Delta N_eff > 5`.
5. Returns `dnnu = NaN` for rejected spectra.
6. Masks the rejected `log10OmegaGW` curve as `NaN`.

Example:

```python
from sagenetgw.classes import GWPredictor
import numpy as np
from matplotlib import pyplot as plt

predictor = GWPredictor(
    model_type="Transformer",
    device="cpu",
)

params = {
    "r":         3.9585109e-05,
    "n_t":       1.0116972,
    "kappa10":   110.42477,
    "T_re":      0.17453859,
    "DN_re":     39.366618,
    "Omega_bh2": 0.0223828,
    "Omega_ch2": 0.1201075,
    "H0":        67.32117,
    "A_s":       2.100549e-9,
}

prediction = predictor.predict_with_dnnu(params)

f = np.asarray(prediction["f"], dtype=float)
log10OmegaGW = np.asarray(prediction["log10OmegaGW"], dtype=float)

print("f sorted:", bool(np.all(np.diff(f) > 0)))
print("spectrum_sorted:", prediction.get("spectrum_sorted"))
print("dnnu_f_mode:", prediction.get("dnnu_f_mode"))

print("dnnu:", prediction["dnnu"])
print("dnnu_raw:", prediction["dnnu_raw"])
print("dnnu_rejected:", prediction["dnnu_rejected"])
print("dnnu_rejected_reason:", prediction["dnnu_rejected_reason"])

if prediction["dnnu_rejected"]:
    print(
        "This spectrum was rejected because Delta N_eff > 5. "
        "The user-facing dnnu is NaN and log10OmegaGW has been masked."
    )
else:
    plt.figure(figsize=(7, 5))
    plt.plot(f, log10OmegaGW, "--", color="royalblue", marker=".")
    plt.xlabel("sorted f or log10(f)")
    plt.ylabel(r"$\log_{10}\Omega_{\rm GW}$")
    plt.title(
        f"SGWB prediction with dnnu={prediction['dnnu']:.6g}"
    )
    plt.tight_layout()
    plt.show()
```

---

## Returned Keys from `predict_with_dnnu()`

The dictionary returned by `predict_with_dnnu()` contains the original spectrum and additional `Delta N_eff` diagnostics:

| Key | Meaning |
|---|---|
| `f` | Sorted frequency grid. For current SageNet models this is usually `log10(f)`. |
| `log10OmegaGW` | Sorted `log10(Omega_GW)` spectrum. If rejected, this is fully masked as `NaN`. |
| `dnnu` | User-facing `Delta N_eff`. If `dnnu_raw > 5`, this is `NaN`. |
| `dnnu_raw` | Raw integrated `Delta N_eff` before the `Delta N_eff > 5` gate. |
| `dnnu_g2` | Integrated SGWB energy-density quantity used to compute `Delta N_eff`. |
| `dnnu_rejected` | Boolean flag indicating whether the sample was rejected. |
| `dnnu_rejected_reason` | Rejection reason. Currently `"dnnu_above_5"` for `Delta N_eff > 5`. |
| `dnnu_diagnostics` | Integration diagnostics, including method, convergence, and grid information. |
| `spectrum_sorted` | Boolean flag showing whether the returned spectrum was sorted. |
| `dnnu_f_mode` | Internal frequency-mode diagnostic, for example `assume_log10f_nonpositive_seen`. |

---

## `Delta N_eff > 5` Guard

The built-in `Delta N_eff` interface uses a fixed non-extrapolation rule:

```text
Delta N_eff > 5  -> rejected
```

When this happens:

```python
prediction["dnnu"] == np.nan
prediction["dnnu_rejected"] == True
prediction["dnnu_rejected_reason"] == "dnnu_above_5"
np.all(np.isnan(prediction["log10OmegaGW"])) == True
```

The raw value is still stored as:

```python
prediction["dnnu_raw"]
```

This allows the user to inspect why the sample was rejected without accidentally using the extrapolated SGWB spectrum as a physical prediction.

---

## Batch Prediction with `Delta N_eff`

For a list of parameter dictionaries, use:

```python
from sagenetgw.classes import GWPredictor

predictor = GWPredictor(
    model_type="Transformer",
    device="cpu",
)

params_list = [
    {
        "r":         3.9585109e-05,
        "n_t":       1.0116972,
        "kappa10":   1.42477,
        "T_re":      0.17453859,
        "DN_re":     39.366618,
        "Omega_bh2": 0.0223828,
        "Omega_ch2": 0.1201075,
        "H0":        67.32117,
        "A_s":       2.100549e-9,
    },
    {
        "r":         1.0e-10,
        "n_t":       2.0,
        "kappa10":   10.0,
        "T_re":      1.0e3,
        "DN_re":     20.0,
        "Omega_bh2": 0.0224,
        "Omega_ch2": 0.12,
        "H0":        67.4,
        "A_s":       2.1e-9,
    },
]

out = predictor.predict_batch_with_dnnu(params_list)

print(out["dnnu"])
print(out["dnnu_raw"])
print(out["dnnu_rejected"])
print(out["dnnu_rejected_reason"])
```

If one sample has `Delta N_eff > 5`, only that sample is rejected. The remaining samples continue normally.

---

## Numerical Backend

SageNetGW also supports the numerical backend:

```python
from sagenetgw.classes import GWPredictor

predictor = GWPredictor(model_type="Numerical")

prediction = predictor.predict({
    "r":         3.9585109e-05,
    "n_t":       1.0116972,
    "kappa10":   1.42477,
    "T_re":      0.17453859,
    "DN_re":     39.366618,
    "Omega_bh2": 0.0223828,
    "Omega_ch2": 0.1201075,
    "H0":        67.32117,
    "A_s":       2.100549e-9,
})
```

For most parameter scans and MCMC applications, the neural-network interface is much faster than direct numerical integration.

---

## Parameter Ranges

The following cosmological parameters are supported:

| Parameter | Range | Scale |
|---|---:|---|
| `r` | `[1e-40, 1]` | Logarithmic |
| `n_t` | `[-1, 6]` | Linear |
| `kappa10` | `[1e-7, 1e3]` | Logarithmic |
| `T_re` | `[1e-3, 1e7] GeV` | Logarithmic |
| `DN_re` | `[0, 40]` | Linear |
| `Omega_bh2` | `[0.005, 0.1]` | Linear |
| `Omega_ch2` | `[0.001, 0.99]` | Linear |
| `H0` | `[20, 100] km/s/Mpc` | Linear |
| `A_s` | `[exp(1.61)/1e10, exp(3.91)/1e10]` | Linear |

The predictor will issue warnings when input parameters are outside the nominal training range.

---

## Notes on Frequency Ordering

The raw `predict()` interface preserves the original model output. The output frequency grid may not be sorted.

The new `predict_with_dnnu()` interface automatically sorts and deduplicates the frequency grid before returning it. Therefore, for plotting and `Delta N_eff` applications, `predict_with_dnnu()` is recommended.

You can check sorting with:

```python
import numpy as np

f = np.asarray(prediction["f"], dtype=float)
print(np.all(np.diff(f) > 0))
```

---

## Citation

If you use SageNetGW or SageNet+ in your research, please cite:

> Zhang F, Luo Y, Li B, et al. (2025).  
> SageNet: Fast Neural Network Emulation of the Stiff-amplified Gravitational Waves from Inflation.  
> *The Astrophysical Journal Supplement Series*, 279(2), 44.  
> doi:10.3847/1538-4365/ade4c6

---

## License

SageNetGW is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
