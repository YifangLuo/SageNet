# SageNetGW
SageNetGW is a Python package for generating GW power-frequency curves.
See https://github.com/YifangLuo/SageNet and https://github.com/bohuarolandli/stiffGWpy

## Installation
```bash
pip install sagenetgw
```
## Usage Example
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