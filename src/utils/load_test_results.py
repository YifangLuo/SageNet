import json
import numpy as np
import os


def load_test_results(output_file="./model_results.json"):
    """
    load (legacy version)
    """
    with open(output_file) as f:
        loaded_json = json.load(f)

    restored_results = {}
    for model_title in loaded_json:
        restored_results[model_title] = {
            'errors_area': np.array(loaded_json[model_title]['errors_area']),
            'errors_smape': np.array(loaded_json[model_title]['errors_smape']),
            'param_values': {k: np.array(v) for k, v in loaded_json[model_title]['param_values'].items()}
        }
    return restored_results


def save(results, filename="model_results.json"):
    """
    save
    """
    output_dir = "./"
    output_file = os.path.join(output_dir, filename)
    with open(output_file, 'w') as f:
        json_results = {}
        for model_title in results:
            json_results[model_title] = {
                'errors_area': results[model_title]['errors_area'].tolist(),
                'errors_smape': results[model_title]['errors_smape'].tolist(),
                'errors_endpoint': results[model_title]['errors_endpoint'].tolist(),
                'param_values': {k: v.tolist() for k, v in results[model_title]['param_values'].items()}
            }
        json.dump(json_results, f, indent=4)

    print(f"Results saved to {output_file}")


def load(filename='model_results.json'):
    """
    load
    """
    with open(filename) as f:
        loaded_json = json.load(f)

    restored_results = {}
    for model_title in loaded_json:
        restored_results[model_title] = {
            'errors_area': np.array(loaded_json[model_title]['errors_area']),
            'errors_smape': np.array(loaded_json[model_title]['errors_smape']),
            'errors_endpoint': np.array(loaded_json[model_title]['errors_endpoint']),
            'param_values': {k: np.array(v) for k, v in loaded_json[model_title]['param_values'].items()}
        }
    return restored_results
