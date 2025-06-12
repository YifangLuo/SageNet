import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy import log10
import warnings

from models import LSTM, Former


class GWDataset(Dataset):
    def __init__(self, data, x_scaler=None, y_scaler=None, param_scaler=None, fit_scalers=True):
        self.data = data

        params = np.array([[log10(item['r']), item['n_t'], log10(item['kappa10']),
                            log10(item['T_re']), item['DN_re'],
                            item['Omega_bh2'], item['Omega_ch2'], item['H0'], item['A_s']] for item in data])
        curves = np.array([np.column_stack((item['f_interp'],
                                            item['log10OmegaGW_interp']))
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
    def __init__(self, model_path='best_gw_model.pth', model_type='Transformer', device='cpu'):
        """
        Initialize the GWPredictor with specified model type and device.

        Args:
            model_path (str): Path to the pretrained model checkpoint.
            model_type (str): Type of model ('LSTM' or 'Transformer').
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        if device not in ['cpu', 'cuda']:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system")

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        if model_type == 'LSTM':
            self.model = LSTM()
        elif model_type == 'Transformer':
            self.model = Former()
        else:
            raise ValueError("model_type must be 'LSTM' or 'Transformer'")

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

        Parameter Descriptions:
        ----------------------
        r : float
            Tensor-to-scalar ratio, logarithmically scaled in the range [1e-40, 1].
        n_t : float
            Tensor spectral index, linearly scaled in the range [-1, 6].
        kappa10 : float
            Curvature perturbation parameter, logarithmically scaled in the range [1e-7, 1e3].
        T_re : float
            Reheating temperature, logarithmically scaled in the range [1e-3, 1e7].
        DN_re : float
            Number of e-folds during reheating, linearly scaled in the range [0, 40].
        Omega_bh2 : float
            Baryon density parameter, in the range [0.005, 0.1].
        Omega_ch2 : float
            Cold dark matter density parameter, in the range [0.001, 0.99].
        H0 : float
            Hubble constant, in the range [20, 100].
        A_s : float
            Scalar amplitude, where ln(10^10 * A_s) is in the range [1.61, 3.91].

        Args:
            params_dict (dict): Dictionary containing parameters:
                r, n_t, kappa10, T_re, DN_re, Omega_bh2, Omega_ch2, H0, A_s

        Returns:
            Dictionary with 'f' (frequency) and 'log10OmegaGW' (GW energy density).

        """
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
            'A_s': (1.61 / (10 ** 10 * np.log(10)), 3.91 / (10 ** 10 * np.log(10)), 'linear')
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
