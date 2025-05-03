import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from numpy import log10

import json
from tqdm import tqdm


def collate_fn(batch):
    params, curves = zip(*batch)
    return torch.stack(params), torch.stack(curves)


class CurvePredictor(nn.Module):
    def __init__(self, num_points=256):
        super().__init__()
        self.num_points = num_points

        self.param_encoder = nn.Sequential(
            nn.Linear(9, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )

        self.position_embed = nn.Embedding(num_points, 256)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        batch_size = x.size(0)
        encoded_params = self.param_encoder(x)
        seq = encoded_params.unsqueeze(1).repeat(1, self.num_points, 1)
        positions = torch.arange(self.num_points, device=x.device).unsqueeze(0)  # [1, N]
        pos_embed = self.position_embed(positions)
        seq += pos_embed
        transformed = self.transformer(seq)
        outputs = self.decoder(transformed)
        return outputs
        # return outputs.permute(0, 2, 1)


class GWPredictor:
    def __init__(self, model_path='best_gw_model.pth'):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        self.model = CurvePredictor()
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

        self.x_scaler = checkpoint['x_scaler']
        self.y_scaler = checkpoint['y_scaler']
        self.param_scaler = checkpoint['param_scaler']

    def predict(self, params_dict):
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
            inputs = torch.tensor(scaled_params, dtype=torch.float32)
            outputs = self.model(inputs).numpy()

        # denorm = self.y_scaler.inverse_transform(
        #     outputs.reshape(-1, 2)).reshape(outputs.shape)
        denorm_x = self.x_scaler.inverse_transform(outputs[..., 0].reshape(-1, 1)).reshape(outputs.shape[0], -1)
        denorm_y = self.y_scaler.inverse_transform(outputs[..., 1].reshape(-1, 1)).reshape(outputs.shape[0], -1)

        return {
            'f': denorm_x[0].tolist(),
            'log10OmegaGW': denorm_y[0].tolist()
        }

# trained_model = train_gw_model("solve_rev.data_solved.json", epochs=200)
