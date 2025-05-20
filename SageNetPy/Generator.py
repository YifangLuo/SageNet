from .classes import GWPredictorFormer
import numpy as np


class FormerGenerator:
    def __init__(self):
        self.model = GWPredictorFormer()

    def generate(self, r: float, n_t: float, kappa10: float, T_re: float, DN_re: float, Omega_bh2: float,
                 Omega_ch2: float, H0: float, A_s: float) -> np.ndarray:
        """
        Generate gravitational wave predictions using the GWPredictorFormer model.

        Parameters
        ----------
        r : float
            Tensor-to-scalar ratio, logarithmically scaled in the range [1e-25, 1].
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

        Returns
        -------
        np.ndarray
            Array of shape (256, 2) containing predicted coordinates. The first column
            is the frequency (f) array of length 256, and the second column is the
            logarithmic gravitational wave amplitude (log10OmegaGW) array of length 256.
        """
        input_params = {
            'r': r,
            'n_t': n_t,
            'kappa10': kappa10,
            'T_re': T_re,
            'DN_re': DN_re,
            'Omega_bh2': Omega_bh2,
            'Omega_ch2': Omega_ch2,
            'H0': H0,
            'A_s': A_s
        }
        prediction = self.model.predict(input_params)
        pred_coords = np.column_stack((prediction['f'], prediction['log10OmegaGW']))
        return pred_coords