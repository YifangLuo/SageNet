import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import seaborn as sns
from .init_plot_style import init_style
init_style(mpl, sns)
# client = pymongo.MongoClient("mongodb://localhost:27017/")
# db = client["solve_rev"]
# collection = db["data_solved_test_2000"]
# predictor = GWPredictor(model_path="best_gw_model_CosmicNet2_154232.pth", model_type="CosmicNet2")
# data = collection.aggregate(
#     [{'$match': {'f': {'$exists': True}, 'f_interp': {'$exists': True}}}, {'$sample': {'size': 6}}])
# data_list = list(data)


def show_curve(data_list, predictor, interp_true_coords=True, error_func=None,
               f_name="f_interp", omega_name="log10OmegaGW_interp", save_dir=None, save_dpi=120):
    # plt.rcParams['xtick.direction'] = 'in'
    # plt.rcParams['ytick.direction'] = 'in'
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    for i, ax in enumerate(axs.flat):
        if i < len(data_list):
            entry = data_list[i]
            param_text = (
                    f"r = {entry['r']:.3e}, "
                    f"n_t = {entry['n_t']:.2f}, "
                    "$k_{10}$" + f" = {entry['kappa10']:.3e}\n"
                                 "$T_{re}/GeV$" + f" = {entry['T_re']:.3e}, "
                                                  "$\Delta N_{{\mathrm{re}}}$" + f" = {entry['DN_re']:.2f}"
            )
            input_params = {
                'r': entry['r'],
                'n_t': entry['n_t'],
                'kappa10': entry['kappa10'],
                'T_re': entry['T_re'],
                'DN_re': entry['DN_re'],
                'Omega_bh2': entry['Omega_bh2'],
                'Omega_ch2': entry['Omega_ch2'],
                'H0': entry['H0'],
                'A_s': entry['A_s']
            }
            prediction = predictor.predict(input_params)
            true_coords = np.column_stack((entry[f_name], entry[omega_name])) \
                if interp_true_coords else np.column_stack((entry['f'], entry['log10OmegaGW']))
            pred_coords = np.column_stack((prediction['f'], prediction['log10OmegaGW']))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel("$\log_{10}(\,f\,/\mathrm{Hz})$")
            ax.set_ylabel("$\Omega_\mathrm{GW}\,(f)$")
            ax.grid(False)
            ax.set_title(f"Sample {i + 1}")
            ax.plot(true_coords[:, 0], true_coords[:, 1], '--', label='Original $\Omega_\mathrm{GW}\,(f)$', color="red",
                    marker='*')
            ax.plot(pred_coords[:, 0], pred_coords[:, 1], '--', label='Predicted Value', color="royalblue", marker='.')
            if error_func is not None:
                errors = error_func(true_coords=true_coords, pred_coords=pred_coords)
                ax.plot(np.cumsum(errors) / np.arange(1, 257),
                        'purple', label='Cumulative Average Error',color = "royalblue")
                ax.fill_between(true_coords[:, 0], true_coords[:, 1] - errors, true_coords[:, 1] + errors,
                                color='gray', alpha=0.15, label='Errors')
                error_text = (f"Mean Rel Area Diff: {errors.mean():.3e}\n"
                              f"Mean Rel Distance: {errors.mean():.3e}")
                ax.set_title(f"Sample {i + 1}:\n{param_text}\n{error_text}")
            else:
                ax.set_title(f"Sample {i + 1}:\n{param_text}")
            ax.legend()
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir, dpi=save_dpi)
    plt.show()
