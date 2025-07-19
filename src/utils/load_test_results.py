import json
import numpy as np


def load_test_results(output_file="./model_results.json"):
    with open(output_file, 'r') as f:
        loaded_json = json.load(f)

    restored_results = {}
    for model_title in loaded_json:
        restored_results[model_title] = {
            'errors_area': np.array(loaded_json[model_title]['errors_area']),
            'errors_smape': np.array(loaded_json[model_title]['errors_smape']),
            'param_values': {k: np.array(v) for k, v in loaded_json[model_title]['param_values'].items()}
        }
    return restored_results
