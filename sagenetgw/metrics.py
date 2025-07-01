import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import quad


def distance(true_coords, pred_coords):
    return np.linalg.norm(true_coords - pred_coords, axis=1)


def calculate_area_difference(true_coords, pred_coords, epsilon=1e-10):
    """
    Calculate the absolute and relative difference of the area under the predicted curve and the true curve
    pred_coords: predicted point coordinates, shape = (n, 2), [f, log10OmegaGW]
    true_coords: true point coordinates, shape = (m, 2), [f, log10OmegaGW]
    Return: (abs_area_diff, rel_area_diff_percent)
    """
    f_true, log10OmegaGW_true = true_coords[:, 0], true_coords[:, 1]
    f_pred, log10OmegaGW_pred = pred_coords[:, 0], pred_coords[:, 1]

    sort_idx_true = np.argsort(f_true)
    f_true = f_true[sort_idx_true]
    log10OmegaGW_true = log10OmegaGW_true[sort_idx_true]

    sort_idx_pred = np.argsort(f_pred)
    f_pred = f_pred[sort_idx_pred]
    log10OmegaGW_pred = log10OmegaGW_pred[sort_idx_pred]

    f_min = max(min(f_true), min(f_pred))
    f_max = min(max(f_true), max(f_pred))
    if f_min >= f_max:
        return float('inf'), float('inf')  # If the ranges do not intersect, an invalid value is returned

    # Interpolation
    t_true = np.linspace(0, 1, len(f_true))
    cs_true = CubicSpline(f_true, log10OmegaGW_true)
    t_pred = np.linspace(0, 1, len(f_pred))
    cs_pred = CubicSpline(f_pred, log10OmegaGW_pred)

    # Calculate the area (take the absolute value)
    area_true, _ = quad(lambda x: abs(cs_true(x)), f_min, f_max)
    area_pred, _ = quad(lambda x: abs(cs_pred(x)), f_min, f_max)
    abs_area_diff = abs(area_pred - area_true)
    rel_area_diff = (abs_area_diff / (area_true + epsilon)) * 100

    return abs_area_diff, rel_area_diff


def calculate_smape(true_coords, pred_coords):
    x_true, y_true = true_coords[:, 0], true_coords[:, 1]
    x_pred, y_pred = pred_coords[:, 0], pred_coords[:, 1]
    smape = 100 * np.mean(2 * np.abs(x_true - x_pred) / np.abs(y_true) + np.abs(y_pred) + 1e-10)
    return smape
