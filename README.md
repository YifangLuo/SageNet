# SageNetGW


## Overview

SageNet+ is an advanced Python package for emulating the stochastic
gravitational wave background (SGWB) spectra from inflation, extending
the SageNet framework described in Zhang et al. (2025). It leverages
deep learning models (LSTM, Transformer, CosmicNet2, or RNN)
and numerical solvers from stiffGWpy to predict the energy density spectrum
with high accuracy and computational efficiency.
SageNet+ supports a wide range of cosmological parameters and achieves a
~10,000-fold speedup over traditional numerical methods.

For more details, see https://github.com/YifangLuo/SageNet 
and https://github.com/bohuarolandli/stiffGWpy

## Installation

SageNet+ is available on PyPI and can be installed using pip:

```bash
pip install sagenetgw
```

### Dependencies

- Python 3.8+
- PyTorch (>=2.0.0)
- NumPy (>=1.20.0)
- scikit-learn (>=1.0.0)

## Quick Start

Below is a simple example to predict an SGWB spectrum using SageNet+:

```python
from sagenetgw.classes import GWPredictor
import numpy as np
from matplotlib import pyplot as plt

predictor = GWPredictor(
        model_type='Transformer',
        device="cpu"
    )

prediction = predictor.predict({
    "r":3.9585109e-05, 
    "n_t":1.0116972, 
    "kappa10":110.42477, 
    "T_re":0.17453859, 
    "DN_re":39.366618,
    "Omega_bh2":0.0223828, 
    "Omega_ch2":0.1201075, 
    "H0":67.32117, 
    "A_s":2.100549e-9
})
pred_coords = np.column_stack((prediction['f'], prediction['log10OmegaGW']))
plt.plot(pred_coords[:, 0], pred_coords[:, 1], '--', color="royalblue", marker='.')
```

Ensure CUDA is installed if using GPU acceleration (by `device='cuda'`).


## Parameter Ranges

The following cosmological parameters are supported:

| Parameter | Range                            | Scale       |
|-----------|----------------------------------|-------------|
| r         | [1e-40, 1]                       | Logarithmic |
| n_t       | [-1, 6]                          | Linear      |
| kappa10   | [1e-7, 1e3]                      | Logarithmic |
| T_re      | [1e-3, 1e7] GeV                  | Logarithmic |
| DN_re     | [0, 40]                          | Linear      |
| Omega_bh2 | [0.005, 0.1]                     | Linear      |
| Omega_ch2 | [0.001, 0.99]                    | Linear      |
| H0        | [20, 100] km/s/Mpc               | Linear      |
| A_s       | [exp(1.61)/1e10, exp(3.91)/1e10] | Linear      |

## Citation

If you use SageNet+ in your research, please cite:

> Zhang, F., Luo, Y., Li, B., et al. (2025). SageNet: Fast Neural Network Emulation of the Stiff-amplified Gravitational
> Waves from Inflation. arXiv:2504.04054.

## License

SageNet+ is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.