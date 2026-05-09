"""import modules"""
import os
import warnings

import numpy as np
import torch
from numpy import log10
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .models import *
from .stiffGWpy.stiff_SGWB import LCDM_SG as sg


_REQUIRED_PARAMS = [
    "r",
    "n_t",
    "kappa10",
    "T_re",
    "DN_re",
    "Omega_bh2",
    "Omega_ch2",
    "H0",
    "A_s",
]


_PARAM_RANGES = {
    "r": (1e-40, 1, "logarithmic"),
    "n_t": (-1, 6, "linear"),
    "kappa10": (1e-7, 1e3, "logarithmic"),
    "T_re": (1e-3, 1e7, "logarithmic"),
    "DN_re": (0, 40, "linear"),
    "Omega_bh2": (0.005, 0.1, "linear"),
    "Omega_ch2": (0.001, 0.99, "linear"),
    "H0": (20, 100, "linear"),
    "A_s": (np.exp(1.61) / 1e10, np.exp(3.91) / 1e10, "linear"),
}


def _check_required_params(params_dict):
    missing_params = [p for p in _REQUIRED_PARAMS if p not in params_dict]
    if missing_params:
        raise KeyError(f"Missing required parameters: {missing_params}")


def _warn_if_outside_training_range(params_dict):
    for param, value in params_dict.items():
        if param not in _PARAM_RANGES:
            continue

        min_val, max_val, scale = _PARAM_RANGES[param]

        if scale == "logarithmic":
            value_to_check = np.log10(value) if value > 0 else float("-inf")
            min_check = np.log10(min_val)
            max_check = np.log10(max_val)
        else:
            value_to_check = value
            min_check = min_val
            max_check = max_val

        if value_to_check < min_check or value_to_check > max_check:
            warnings.warn(
                f"Parameter '{param}' value {value} is outside the valid range "
                f"[{min_val}, {max_val}] "
                f"({'logarithmic' if scale == 'logarithmic' else 'linear'} scale)"
            )


def _stack_if_rectangular(values):
    try:
        arr = np.asarray(values, dtype=float)
        if arr.dtype != object:
            return arr
    except Exception:
        pass
    return values


def _mask_extrapolated_curve(prediction):
    """
    Remove the meaningless extrapolated SGWB curve from the returned prediction.

    The frequency grid is kept, but log10OmegaGW values are replaced with NaN,
    so users cannot accidentally plot or save the rejected curve as a physical
    spectrum.
    """
    if "log10OmegaGW" not in prediction:
        return prediction

    y = np.asarray(prediction["log10OmegaGW"], dtype=float).copy()
    y[...] = np.nan
    prediction["log10OmegaGW"] = y.tolist()

    return prediction


class Numerical:
    """Numerical solver for computing SGWB spectra using stiffGWpy."""

    def __init__(self):
        self.model = None
        return

    def solve(self, params_dict):
        _check_required_params(params_dict)

        self.model = sg(
            r=params_dict["r"],
            n_t=params_dict["n_t"],
            kappa10=params_dict["kappa10"],
            T_re=params_dict["T_re"],
            DN_re=params_dict["DN_re"],
            Omega_bh2=params_dict["Omega_bh2"],
            Omega_ch2=params_dict["Omega_ch2"],
            H0=params_dict["H0"],
            A_s=params_dict["A_s"],
        )

        self.model.SGWB_iter()
        return self.model.f, self.model.log10OmegaGW


class GWDataset(Dataset):
    def __init__(
        self,
        data,
        x_scaler=None,
        y_scaler=None,
        param_scaler=None,
        fit_scalers=True,
        interp_percent=60,
    ):
        self.data = data

        params = np.array(
            [
                [
                    log10(item["r"]),
                    item["n_t"],
                    log10(item["kappa10"]),
                    log10(item["T_re"]),
                    item["DN_re"],
                    item["Omega_bh2"],
                    item["Omega_ch2"],
                    item["H0"],
                    item["A_s"],
                ]
                for item in data
            ]
        )

        curves = np.array(
            [
                np.column_stack(
                    (
                        item[f"f_interp_{interp_percent}"],
                        item[f"log10OmegaGW_interp_{interp_percent}"],
                    )
                )
                for item in data
            ]
        )

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

        curves_x_scaled = self.x_scaler.transform(
            curves_x.reshape(-1, 1)
        ).reshape(curves_x.shape)

        curves_y_scaled = self.y_scaler.transform(
            curves_y.reshape(-1, 1)
        ).reshape(curves_y.shape)

        self.curves = np.stack([curves_x_scaled, curves_y_scaled], axis=2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        params = torch.tensor(self.params[idx], dtype=torch.float32)
        curve = torch.tensor(self.curves[idx], dtype=torch.float32)
        return params, curve


class GWPredictor:
    """Neural network-based predictor for SGWB spectra."""

    def __init__(self, model_path=None, model_type="Transformer", device="cpu"):
        self.model_type = model_type

        if model_type == "Numerical":
            self.solver = Numerical()
            return

        if device not in ["cpu", "cuda"]:
            raise ValueError("device must be 'cpu' or 'cuda'")

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system")

        if model_type == "LSTM":
            self.model = LSTM()
        elif model_type == "Transformer":
            self.model = Former()
        elif model_type == "CosmicNet2":
            self.model = CosmicNet2()
        elif model_type == "RNN":
            self.model = RNN()
        elif model_type == "CNN":
            self.model = CNN()
        elif model_type == "TCN":
            self.model = TCN()
        elif model_type == "GRU":
            self.model = GRU()
        else:
            raise ValueError(
                "model_type must be 'LSTM', 'Transformer', 'CosmicNet2', "
                "'RNN', 'CNN', 'TCN', 'GRU', or 'Numerical'"
            )

        if model_path is not None:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
            checkpoint = torch.load(
                model_path,
                map_location=device,
                weights_only=False,
            )
        else:
            model_dir = os.path.join(os.path.dirname(__file__), "models")
            model_path = os.path.join(model_dir, f"best_gw_model_{model_type}.pth")

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Default model checkpoint not found at {model_path}"
                )

            checkpoint = torch.load(
                model_path,
                map_location=device,
                weights_only=False,
            )

        self.model.load_state_dict(checkpoint["model_state"])
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.x_scaler = checkpoint["x_scaler"]
        self.y_scaler = checkpoint["y_scaler"]
        self.param_scaler = checkpoint["param_scaler"]

    def predict_value(self, params_dict):
        """
        Predict gravitational wave signal based on input parameters.

        This method preserves the original SageNet interface and returns only:
            - f
            - log10OmegaGW
        """
        _check_required_params(params_dict)
        _warn_if_outside_training_range(params_dict)

        if self.model_type == "Numerical":
            f_solved, omega_solved = self.solver.solve(params_dict)
            return {
                "f": np.asarray(f_solved, dtype=float).tolist(),
                "log10OmegaGW": np.asarray(omega_solved, dtype=float).tolist(),
            }

        params = np.array(
            [
                log10(params_dict["r"]),
                params_dict["n_t"],
                log10(params_dict["kappa10"]),
                log10(params_dict["T_re"]),
                params_dict["DN_re"],
                params_dict["Omega_bh2"],
                params_dict["Omega_ch2"],
                params_dict["H0"],
                params_dict["A_s"],
            ]
        ).reshape(1, -1)

        scaled_params = self.param_scaler.transform(params)

        with torch.no_grad():
            inputs = torch.tensor(scaled_params, dtype=torch.float32).to(self.device)
            outputs = self.model(inputs).to("cpu").numpy()

        denorm_x = self.x_scaler.inverse_transform(
            outputs[..., 0].reshape(-1, 1)
        ).reshape(outputs.shape[0], -1)

        denorm_y = self.y_scaler.inverse_transform(
            outputs[..., 1].reshape(-1, 1)
        ).reshape(outputs.shape[0], -1)

        return {
            "f": denorm_x[0].tolist(),
            "log10OmegaGW": denorm_y[0].tolist(),
        }

    def predict(self, params_dict, _sample_index=None):
        """
        Predict SGWB spectrum and compute Delta N_eff.

        Internal automatic processing:
            predict()
            -> sort/clean/unique spectrum
            -> compute delta_N_eff
            -> if Delta N_eff > 5, delta_N_eff=NaN and log10OmegaGW=NaN

        Users do not need to call any sorting or delta_N_eff helper manually.
        """
        _check_required_params(params_dict)

        prediction = self.predict_value(params_dict)

        from .delta_N_eff import compute_delta_N_eff
        from .delta_N_eff.api import _sort_prediction_spectrum_for_output

        prediction = _sort_prediction_spectrum_for_output(prediction)

        result = compute_delta_N_eff(
            prediction,
            H0=float(params_dict["H0"]),
            sample_index=_sample_index,
        )

        prediction["delta_N_eff"] = float(result.delta_N_eff)
        prediction["delta_N_eff_raw"] = float(result.delta_N_eff_raw)
        prediction["delta_N_eff_g2"] = float(result.g2)
        prediction["delta_N_eff_rejected"] = bool(result.rejected)
        prediction["delta_N_eff_rejected_reason"] = result.rejected_reason
        prediction["delta_N_eff_diagnostics"] = result.diagnostics

        if result.rejected and result.rejected_reason == "delta_N_eff_above_5":
            prediction = _mask_extrapolated_curve(prediction)

        return prediction

    def predict_batch_with_delta_N_eff(self, params_list):
        """
        Predict SGWB spectra and Delta N_eff for a list of parameter dictionaries.

        Batch-safe behaviour:
            if one sample has Delta N_eff > 5, only that sample is marked
            invalid; the remaining samples continue.
        """
        if not isinstance(params_list, (list, tuple)):
            raise TypeError(
                "predict_batch_with_delta_N_eff expects a list or tuple of parameter dictionaries."
            )

        predictions = []

        for i, params_dict in enumerate(params_list):
            prediction_i = self.predict(
                params_dict,
                _sample_index=i,
            )
            predictions.append(prediction_i)

        f_values = [p["f"] for p in predictions]
        y_values = [p["log10OmegaGW"] for p in predictions]

        return {
            "f": _stack_if_rectangular(f_values),
            "log10OmegaGW": _stack_if_rectangular(y_values),
            "delta_N_eff": np.asarray(
                [p["delta_N_eff"] for p in predictions],
                dtype=float,
            ),
            "delta_N_eff_raw": np.asarray(
                [p["delta_N_eff_raw"] for p in predictions],
                dtype=float,
            ),
            "delta_N_eff_g2": np.asarray(
                [p["delta_N_eff_g2"] for p in predictions],
                dtype=float,
            ),
            "delta_N_eff_rejected": np.asarray(
                [p["delta_N_eff_rejected"] for p in predictions],
                dtype=bool,
            ),
            "delta_N_eff_rejected_reason": np.asarray(
                [p["delta_N_eff_rejected_reason"] for p in predictions],
                dtype=object,
            ),
            "delta_N_eff_diagnostics": [
                p["delta_N_eff_diagnostics"] for p in predictions
            ],
            "predictions": predictions,
        }
