"""import modules"""
import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy import log10
import warnings

from .models import LSTM, Former, CosmicNet2, RNN, CNN, TCN
from .stiffGWpy.stiff_SGWB import LCDM_SG as sg


class Numerical:
    """Numerical solver for computing SGWB spectra using stiffGWpy.
    This class interfaces with the stiffGWpy package to numerically solve the tensor
    wave equations for the inflationary SGWB with stiff amplification.
    See https://github.com/bohuarolandli/stiffGWpy for more details.
    """

    def __init__(self):
        """Initialize a Numerical solver."""
        self.model = None
        return

    def solve(self, params_dict):
        """call the numerical methods and compute SGWB spectrum numerically, same as stiffGWpy.
        Args:
                params_dict (dict): Dictionary containing cosmological parameters:

                    - r: Tensor-to-scalar ratio (logarithmic scale, [1e-40, 1]).
                    - n_t: Tensor spectral index (linear scale, [-1, 6]).
                    - kappa10: Curvature perturbation parameter (logarithmic scale, [1e-7, 1e3]).
                    - T_re: Reheating temperature (logarithmic scale, [1e-3, 1e7] GeV).
                    - DN_re: Number of e-folds during reheating (linear scale, [0, 40]).
                    - Omega_bh2: Baryon density parameter ([0.005, 0.1]).
                    - Omega_ch2: Cold dark matter density parameter ([0.001, 0.99]).
                    - H0: Hubble constant ([20, 100] km/s/Mpc).
                    - A_s: Scalar amplitude (exp(ln(10^10 * A_s)) in [1.61, 3.91]).
        Returns:
                tuple: (frequencies, log10OmegaGW) where frequencies is a numpy array of
                    sampled frequencies and log10OmegaGW is the logarithmic SGWB spectrum.

        Raises:
                KeyError: If required parameters are missing in params_dict.
        """
        required_params = ['r', 'n_t', 'kappa10', 'T_re', 'DN_re', 'Omega_bh2', 'Omega_ch2', 'H0', 'A_s']
        missing_params = [p for p in required_params if p not in params_dict]
        if missing_params:
            raise KeyError(f"Missing required parameters: {missing_params}")

        self.model = sg(
            r=params_dict['r'],
            n_t=params_dict['n_t'],
            kappa10=params_dict['kappa10'],
            T_re=params_dict['T_re'],
            DN_re=params_dict['DN_re'],
            Omega_bh2=params_dict['Omega_bh2'],
            Omega_ch2=params_dict['Omega_ch2'],
            H0=params_dict['H0'],
            A_s=params_dict['A_s']
        )

        self.model.SGWB_iter()  # Perform iterative numerical integration
        return self.model.f, self.model.log10OmegaGW


class GWDataset(Dataset):
    def __init__(self, data, x_scaler=None, y_scaler=None, param_scaler=None, fit_scalers=True, interp_percent=60):
        self.data = data

        params = np.array([[log10(item['r']), item['n_t'], log10(item['kappa10']),
                            log10(item['T_re']), item['DN_re'],
                            item['Omega_bh2'], item['Omega_ch2'], item['H0'], item['A_s']] for item in data])
        curves = np.array([np.column_stack((item[f'f_interp_{interp_percent}'],
                                            item[f'log10OmegaGW_interp_{interp_percent}']))
                           for item in data])

        # split x and y
        curves_x = curves[:, :, 0]
        curves_y = curves[:, :, 1]

        if fit_scalers or x_scaler or y_scaler or param_scaler is None:
            self.param_scaler = StandardScaler()
            self.param_scaler.fit(params)
            self.x_scaler = StandardScaler()
            self.x_scaler.fit(curves_x.reshape(-1, 1))
            self.y_scaler = StandardScaler()
            self.y_scaler.fit(curves_y.reshape(-1, 1))
        else:
            self.param_scaler = param_scaler
            self.x_scaler = x_scaler
            self.y_scaler = y_scaler

        self.params = self.param_scaler.transform(params)
        curves_x_scaled = self.x_scaler.transform(curves_x.reshape(-1, 1)).reshape(curves_x.shape)
        curves_y_scaled = self.y_scaler.transform(curves_y.reshape(-1, 1)).reshape(curves_y.shape)
        self.curves = np.stack([curves_x_scaled, curves_y_scaled], axis=2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        params = torch.tensor(self.params[idx], dtype=torch.float32)
        curve = torch.tensor(self.curves[idx], dtype=torch.float32)
        return params, curve


class GWPredictor:
    """Neural network-based predictor for SGWB spectra.
    This class loads a pretrained neural network model (Transformer, LSTM, RNN or CosmicNet2)
    or uses a numerical solver to predict the SGWB spectrum based on cosmological parameters.
    """

    def __init__(self, model_path=None, model_type='Transformer', device='cpu'):
        """Initialize the GWPredictor with specified model type and device.

        Args:
            model_path (str, optional): Path to the pretrained model checkpoint. If None,
                loads the default model for the specified model_type.
            model_type (str): Type of model ('LSTM', 'Transformer', 'CosmicNet2', 'RNN', or 'Numerical').
            device (str): Device to run the model on ('cpu' or 'cuda').

        Raises:
            ValueError: If model_type or device is invalid.
            RuntimeError: If CUDA is specified but unavailable.
            FileNotFoundError: If model_path or default model file does not exist.
        """
        self.model_type = model_type
        if model_type == 'Numerical':
            self.solver = Numerical()
            return

        if device not in ['cpu', 'cuda']:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system")

        # Initialize model based on type
        if model_type == 'LSTM':
            self.model = LSTM()
        elif model_type == 'Transformer':
            self.model = Former()
        elif model_type == 'CosmicNet2':
            self.model = CosmicNet2()
        elif model_type == 'RNN':
            self.model = RNN()
        elif model_type == 'CNN':
            self.model = CNN()
        elif model_type == 'TCN':
            self.model = TCN()
        else:
            raise ValueError("model_type must be 'LSTM', 'Transformer', 'CosmicNet2', 'RNN', or 'Numerical'")

        if model_path is not None:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        else:
            model_dir = os.path.join(os.path.dirname(__file__), 'models')
            model_path = os.path.join(model_dir, f'best_gw_model_{model_type}.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Default model checkpoint not found at {model_path}")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state'])
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.x_scaler = checkpoint['x_scaler']
        self.y_scaler = checkpoint['y_scaler']
        self.param_scaler = checkpoint['param_scaler']

    def predict(self, params_dict):
        """
        Predict gravitational wave signal based on input parameters.

        Args:
            params_dict (dict): Dictionary containing cosmological parameters:

                - r: Tensor-to-scalar ratio (logarithmic scale, [1e-40, 1]).
                - n_t: Tensor spectral index (linear scale, [-1, 6]).
                - kappa10: Curvature perturbation parameter (logarithmic scale, [1e-7, 1e3]).
                - T_re: Reheating temperature (logarithmic scale, [1e-3, 1e7] GeV).
                - DN_re: Number of e-folds during reheating (linear scale, [0, 40]).
                - Omega_bh2: Baryon density parameter ([0.005, 0.1]).
                - Omega_ch2: Cold dark matter density parameter ([0.001, 0.99]).
                - H0: Hubble constant ([20, 100] km/s/Mpc).
                - A_s: Scalar amplitude (exp(ln(10^10 * A_s)) in [1.61, 3.91]).

        Returns:
            dict: Dictionary with keys:
                - 'f': List of frequency values.
                - 'log10OmegaGW': List of logarithmic SGWB spectrum values.

        Raises:
            KeyError: If required parameters are missing in params_dict.
            ValueError: If parameter values are invalid (e.g., non-positive for logarithmic scale).
        """

        required_params = ['r', 'n_t', 'kappa10', 'T_re', 'DN_re', 'Omega_bh2', 'Omega_ch2', 'H0', 'A_s']
        missing_params = [p for p in required_params if p not in params_dict]
        if missing_params:
            raise KeyError(f"Missing required parameters: {missing_params}")

        param_ranges = {
            'r': (1e-40, 1, 'logarithmic'),
            'n_t': (-1, 6, 'linear'),
            'kappa10': (1e-7, 1e3, 'logarithmic'),
            'T_re': (1e-3, 1e7, 'logarithmic'),
            'DN_re': (0, 40, 'linear'),
            'Omega_bh2': (0.005, 0.1, 'linear'),
            'Omega_ch2': (0.001, 0.99, 'linear'),
            'H0': (20, 100, 'linear'),
            # Convert ln(10^10 * A_s) range to A_s range
            'A_s': (np.exp(1.61) / 1e10, np.exp(3.91) / 1e10, 'linear')
        }

        for param, value in params_dict.items():
            if param not in param_ranges:
                continue
            min_val, max_val, scale = param_ranges[param]
            if scale == 'logarithmic':
                value_to_check = np.log10(value) if value > 0 else float('-inf')
                min_val = np.log10(min_val)
                max_val = np.log10(max_val)
            else:
                value_to_check = value

            if value_to_check < min_val or value_to_check > max_val:
                warnings.warn(
                    f"Parameter '{param}' value {value} is outside the valid range "
                    f"[{param_ranges[param][0]}, {param_ranges[param][1]}] "
                    f"({'logarithmic' if param_ranges[param][2] == 'logarithmic' else 'linear'} scale)"
                )

        if self.model_type == "Numerical":
            f_solved, omega_solved = self.solver.solve(params_dict)
            return {
                'f': f_solved,
                'log10OmegaGW': omega_solved.tolist()
            }
        else:
            params = np.array([
                log10(params_dict['r']),
                params_dict['n_t'],
                log10(params_dict['kappa10']),
                log10(params_dict['T_re']),
                params_dict['DN_re'],
                params_dict['Omega_bh2'],
                params_dict['Omega_ch2'],
                params_dict['H0'],
                params_dict['A_s']
            ]).reshape(1, -1)
            scaled_params = self.param_scaler.transform(params)

            with torch.no_grad():
                inputs = torch.tensor(scaled_params, dtype=torch.float32).to(self.device)
                outputs = self.model(inputs).to('cpu').numpy()

            denorm_x = self.x_scaler.inverse_transform(outputs[..., 0].reshape(-1, 1)).reshape(outputs.shape[0], -1)
            denorm_y = self.y_scaler.inverse_transform(outputs[..., 1].reshape(-1, 1)).reshape(outputs.shape[0], -1)

            return {
                'f': denorm_x[0].tolist(),
                'log10OmegaGW': denorm_y[0].tolist()
            }
