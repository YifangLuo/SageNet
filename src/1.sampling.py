import json
import os
import math
import numpy as np
from pyDOE import lhs
from numpy.typing import NDArray


def generate_parameters_lhs(num_samples: int, output_file="lhs_samples.json") -> NDArray[np.float64]:
    """
    Generate parameter samples using Latin Hypercube Sampling (LHS) based on pyDOE.

    Args:
        num_samples (int): Number of samples to generate.
        output_file (str): Name of output.

    Returns:
        NDArray[np.float64]: Array of shape (num_samples, 9) with columns in the order
                            [r, n_t, kappa10, T_re, DN_re, Omega_bh2, Omega_ch2, H0, A_s].
    """
    # Generate LHS samples in range [0, 1)
    lhs_samples = lhs(9, samples=num_samples, criterion='maximin')

    # Scale each parameter
    # r: logarithmic [1e-25, 1]
    log_r = np.log10(1e-25) + lhs_samples[:, 0] * (np.log10(1) - np.log10(1e-25))
    r = 10 ** log_r
    # n_t: linear [-1, 6]
    n_t = lhs_samples[:, 1] * (6 - (-1)) + (-1)
    # kappa10: logarithmic [1e-7, 1e3]
    log_kappa = np.log10(1e-7) + lhs_samples[:, 2] * (np.log10(1e3) - np.log10(1e-7))
    kappa10 = 10 ** log_kappa
    # T_re: linear [1e-3, 1e7]
    log_T = np.log10(1e-3) + lhs_samples[:, 3] * (np.log10(1e7) - np.log10(1e-3))
    T_re = 10 ** log_T
    # DN_re: linear [0, 40]
    DN_re = lhs_samples[:, 4] * 40
    # Omega_bh2: linear [0.014, 0.03] to [0.005, 0.1]
    Omega_bh2 = lhs_samples[:, 5] * (0.1 - 0.005) + 0.005
    # Omega_ch2: linear [0.1, 0.14] to [0.001, 0.99]
    Omega_ch2 = lhs_samples[:, 6] * (0.99 - 0.001) + 0.001
    # H0: linear [50, 85] to [20,100]
    H0 = lhs_samples[:, 7] * (100 - 20) + 20
    # ln(10^10 * A_s): linear [2.8, 3.2] to [1.61, 3.91]
    ln_1e10_A_s = lhs_samples[:, 8] * (3.91 - 1.61) + 1.61
    A_s = np.exp(ln_1e10_A_s) / 1e10

    result = np.column_stack((r, n_t, kappa10, T_re, DN_re, Omega_bh2, Omega_ch2, H0, A_s))
    param_names = ["r", "n_t", "kappa10", "T_re", "DN_re", "Omega_bh2", "Omega_ch2", "H0", "A_s"]
    json_data = [
        {param_names[i]: float(result[j, i]) for i in range(len(param_names))}
        for j in range(num_samples)
    ]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

    return result


def split_json_file(input_file, output_dir, num_parts):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Calculate samples per part
    total_samples = len(data)
    samples_per_part = math.ceil(total_samples / num_parts)

    # Split and save each part
    for i in range(num_parts):
        start_idx = i * samples_per_part
        end_idx = min((i + 1) * samples_per_part, total_samples)

        # Get the chunk of data
        chunk = data[start_idx:end_idx]

        # Define output file path
        output_file = os.path.join(output_dir, f'data_part_{i + 1}.json')

        # Write chunk to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)

        print(f'Saved {len(chunk)} samples to {output_file}')


if __name__ == "__main__":
    raw_file = "sagenet_plus_30000.json"
    print("Start LHS sampling.")
    samples = generate_parameters_lhs(30000, output_file=raw_file)
    print("Samples generated and saved.")
    output_dir = "data_part"
    num_parts = 3
    split_json_file(raw_file, output_dir, num_parts)


