"""
Metrics.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import quad, simpson
from scipy.interpolate import PchipInterpolator
import numpy as np
import warnings


def distance(true_coords, pred_coords):
    """
    Simple distance of points. Cause large errors.
    """
    return np.linalg.norm(true_coords - pred_coords, axis=1)


def sort_coords(true_coords, pred_coords):
    """
    Sort true and predicted coordinates by frequency (f) in ascending order,
    ensuring f_pred is strictly monotonically increasing by adding small perturbations
    to duplicate frequencies while preserving sequence length.
    """
    # Extract frequencies and log10OmegaGW
    f_true, log10OmegaGW_true = true_coords[:, 0], true_coords[:, 1]
    f_pred, log10OmegaGW_pred = pred_coords[:, 0], pred_coords[:, 1]

    # Sort true_coords
    sort_idx_true = np.argsort(f_true)
    f_true = f_true[sort_idx_true]
    log10OmegaGW_true = log10OmegaGW_true[sort_idx_true]

    # Sort pred_coords and handle duplicates in f_pred
    sort_idx_pred = np.argsort(f_pred)
    f_pred = f_pred[sort_idx_pred]
    log10OmegaGW_pred = log10OmegaGW_pred[sort_idx_pred]

    # Check for duplicates in f_pred and add small perturbations
    if len(f_pred) > 1:
        diff = np.diff(f_pred)
        if not np.all(diff > 0):
            # Identify duplicate values
            epsilon = 1e-10 * (f_pred.max() - f_pred.min())  # Small perturbation relative to range
            if epsilon == 0:  # Handle case where all frequencies are the same
                epsilon = 1e-10
            # Add increasing perturbations to ensure strict monotonicity
            for i in range(1, len(f_pred)):
                if f_pred[i] <= f_pred[i - 1]:
                    f_pred[i:] += epsilon * (i + 1 - np.arange(i, len(f_pred)))

    # Verify strict monotonicity
    if len(f_pred) > 1 and not np.all(np.diff(f_pred) > 0):
        warnings.warn("Warning: Failed to make f_pred strictly monotonically increasing.")

    return f_true, log10OmegaGW_true, f_pred, log10OmegaGW_pred


def calculate_area_difference(true_coords, pred_coords):
    """
    Calculate the absolute and relative difference of the area under the predicted curve and the true curve
    pred_coords: predicted point coordinates, shape = (n, 2), [f, log10OmegaGW]
    true_coords: true point coordinates, shape = (m, 2), [f, log10OmegaGW]
    Return: (abs_area_diff, rel_area_diff_percent)
    """

    f_true, log10OmegaGW_true, f_pred, log10OmegaGW_pred = sort_coords(true_coords=true_coords, pred_coords=pred_coords)

    f_min = max(min(f_true), min(f_pred))
    f_max = min(max(f_true), max(f_pred))
    if f_min >= f_max:
        warnings.warn("Warning: Detected f_min >= f_max. This implies there is a invalid curve.")
        return float('inf'), float('inf')  # If the ranges do not intersect, an invalid value is returned

    # Interpolation. We discarded Cubic because oscillation.
    # cs_true = CubicSpline(f_true, log10OmegaGW_true)
    cs_true = PchipInterpolator(f_true, log10OmegaGW_true)
    # cs_pred = CubicSpline(f_pred, log10OmegaGW_pred)
    cs_pred = PchipInterpolator(f_pred, log10OmegaGW_pred)

    f_grid = np.linspace(f_min, f_max, 10000)
    log10OmegaGW_true_grid = cs_true(f_grid)
    log10OmegaGW_pred_grid = cs_pred(f_grid)

    # Because log10OmegaGW is logarithmic, the difference is actually the OmegaGW ratio.
    abs_area_diff = simpson(y=np.abs(log10OmegaGW_true_grid - log10OmegaGW_pred_grid), x=f_grid, dx=0.001)

    return np.log(10) * abs_area_diff / (f_max - f_min)


def calculate_endpoints(true_coords, pred_coords):
    """
    Endpoint errors.
    """
    f_true, log10OmegaGW_true, f_pred, log10OmegaGW_pred = sort_coords(true_coords=true_coords, pred_coords=pred_coords)
    # true_len = max(f_true) - min(f_true)
    endpoints_error = abs(min(f_true) - min(f_pred)) + abs(max(f_true) - max(f_pred))
    return endpoints_error * np.log(10)


# NOTE: This metrics is not suitable because the log10omegaGW value is logarithmic.
# Simple integration of logarithmic values will lead to the lack of physical significance.
# Now it is deprecated, but we kept it for reference in further work.
# noinspection PyTupleAssignmentBalance
def calculate_area_difference_legacy(true_coords, pred_coords, epsilon=1e-10):
    """
    Calculate the absolute and relative difference of the area under the predicted curve and the true curve
    pred_coords: predicted point coordinates, shape = (n, 2), [f, log10OmegaGW]
    true_coords: true point coordinates, shape = (m, 2), [f, log10OmegaGW]
    Return: (abs_area_diff, rel_area_diff_percent)
    """
    f_true, log10OmegaGW_true = true_coords[:, 0], true_coords[:, 1]
    f_pred, log10OmegaGW_pred = pred_coords[:, 0], pred_coords[:, 1]

    # Convert log10OmegaGW to OmegaGW
    OmegaGW_true = 10 ** log10OmegaGW_true
    OmegaGW_pred = 10 ** log10OmegaGW_pred

    sort_idx_true = np.argsort(f_true)
    f_true = f_true[sort_idx_true]
    OmegaGW_true = OmegaGW_true[sort_idx_true]

    sort_idx_pred = np.argsort(f_pred)
    f_pred = f_pred[sort_idx_pred]
    OmegaGW_pred = OmegaGW_pred[sort_idx_pred]

    f_min = max(min(f_true), min(f_pred))
    f_max = min(max(f_true), max(f_pred))
    if f_min >= f_max:
        return float('inf'), float('inf')  # If the ranges do not intersect, an invalid value is returned

    # Interpolation
    # t_true = np.linspace(0, 1, len(f_true))
    cs_true = CubicSpline(f_true, OmegaGW_true)
    # t_pred = np.linspace(0, 1, len(f_pred))
    cs_pred = CubicSpline(f_pred, OmegaGW_pred)

    # Calculate the area (take the absolute value)
    area_true, _ = quad(lambda x: abs(cs_true(x)), f_min, f_max)
    area_pred, _ = quad(lambda x: abs(cs_pred(x)), f_min, f_max)

    addition = 0
    if min(f_true) < min(f_pred):
        add_, _ = quad(lambda x: abs(cs_true(x)), min(f_true), min(f_pred))
        addition = addition + add_ / area_true
    else:
        add_, _ = quad(lambda x: abs(cs_pred(x)), min(f_pred), min(f_true))
        addition = addition + add_ / area_pred
    if max(f_true) < max(f_pred):
        add_, _ = quad(lambda x: abs(cs_pred(x)), max(f_true), max(f_pred))
        addition = addition + add_ / area_pred
    else:
        add_, _ = quad(lambda x: abs(cs_true(x)), max(f_pred), max(f_true))
        addition = addition + add_ / area_true

    abs_area_diff, _ = quad(lambda x: abs(cs_true(x) - cs_pred(x)), f_min, f_max)
    rel_area_diff = ((abs_area_diff / (area_true + epsilon)) + addition) * 100

    return abs_area_diff, rel_area_diff


def calculate_smape(true_coords, pred_coords):
    """
    A simple metric of SMAPE.
    """
    f_true, log10OmegaGW_true, f_pred, log10OmegaGW_pred = sort_coords(true_coords=true_coords, pred_coords=pred_coords)

    smape = 100 * np.mean(2 * np.abs(log10OmegaGW_pred - log10OmegaGW_true) /
                          (np.abs(log10OmegaGW_true) + np.abs(log10OmegaGW_true) + 1e-10))

    return smape

def calculate_difference_grid(true_coords, pred_coords):
    """
    Calculate the absolute error distribution between the predicted and true curves at each frequency point.
    pred_coords: predicted point coordinates, shape = (n, 2), [f, log10OmegaGW]
    true_coords: true point coordinates, shape = (m, 2), [f, log10OmegaGW]
    Return: (f_grid, abs_error_grid)
        - f_grid: array of frequency points, shape = (10000,)
        - abs_error_grid: array of absolute differences |log10OmegaGW_true - log10OmegaGW_pred|, shape = (10000,)
    """
    f_true, log10OmegaGW_true, f_pred, log10OmegaGW_pred = sort_coords(true_coords=true_coords, pred_coords=pred_coords)

    f_min = max(min(f_true), min(f_pred))
    f_max = min(max(f_true), max(f_pred))
    if f_min >= f_max:
        warnings.warn("Warning: Detected f_min >= f_max. This implies there is an invalid curve.")
        return np.array([]), np.array([])  # Return empty arrays for invalid cases

    # Interpolation using PchipInterpolator
    cs_true = PchipInterpolator(f_true, log10OmegaGW_true)
    cs_pred = PchipInterpolator(f_pred, log10OmegaGW_pred)

    # Create fine grid for evaluation
    f_grid = np.linspace(f_min, f_max, 10000)
    log10OmegaGW_true_grid = cs_true(f_grid)
    log10OmegaGW_pred_grid = cs_pred(f_grid)

    # Calculate absolute error at each grid point
    abs_error_grid = np.abs(log10OmegaGW_true_grid - log10OmegaGW_pred_grid)

    return f_grid, abs_error_grid
