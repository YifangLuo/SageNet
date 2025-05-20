import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from numpy import log10


class GWDataset(Dataset):
    def __init__(self, data, x_scaler=None, y_scaler=None, param_scaler=None, fit_scalers=True):
        self.data = data

        params = np.array([[log10(item['r']), item['n_t'], log10(item['kappa10']),
                            log10(item['T_re']), item['DN_re'],
                            item['Omega_bh2'], item['Omega_ch2'], item['H0'], item['A_s']] for item in data])
        curves = np.array([np.column_stack((item['f_interp'],
                                            item['log10OmegaGW_interp']))
                           for item in data])

        # 分割x和y
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


def collate_fn(batch):
    params, curves = zip(*batch)
    return torch.stack(params), torch.stack(curves)


class CurvePredictorFormer(nn.Module):
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


from tqdm import tqdm


class GWPredictorFormer:
    def __init__(self, model_path='best_gw_model_LSTM.pth'):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        self.model = CurvePredictorFormer()
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

class CurvePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # 参数编码器
        self.encoder = nn.Sequential(
            nn.Linear(5, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            bidirectional=False,
            batch_first=True
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # 编码参数 [B,5] -> [B,256]
        encoded = self.encoder(x)

        # 扩展为序列 [B,256] -> [B,256,256]
        repeated = encoded.unsqueeze(1).repeat(1, 256, 1)

        # 双向LSTM处理 [B,256,256] -> [B,256,512]
        lstm_out, _ = self.lstm(repeated)

        # 解码输出 [B,256,512] -> [B,256,2]
        return self.decoder(lstm_out)

class GWPredictor:
    def __init__(self, model_path='best_gw_model.pth'):
        checkpoint = torch.load(model_path, map_location='cpu',weights_only=False)

        self.model = CurvePredictor()
        self.model.load_state_dict(checkpoint['model_state'])
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        self.model = self.model.to(self.device)
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
            params_dict['DN_re']
        ]).reshape(1, -1)

        scaled_params = self.param_scaler.transform(params)

        with torch.no_grad():
            inputs = torch.tensor(scaled_params, dtype=torch.float32)
            inputs = inputs.to(self.device)
            outputs = self.model(inputs).to('cpu').numpy()

        # denorm = self.y_scaler.inverse_transform(
        #     outputs.reshape(-1, 2)).reshape(outputs.shape)
        denorm_x = self.x_scaler.inverse_transform(outputs[..., 0].reshape(-1, 1)).reshape(outputs.shape[0], -1)
        denorm_y = self.y_scaler.inverse_transform(outputs[..., 1].reshape(-1, 1)).reshape(outputs.shape[0], -1)

        return {
            'f': denorm_x[0].tolist(),
            'log10OmegaGW': denorm_y[0].tolist()
        }

