import torch
import torch.nn as nn


class Former(nn.Module):
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
        # batch_size = x.size(0)
        encoded_params = self.param_encoder(x)
        seq = encoded_params.unsqueeze(1).repeat(1, self.num_points, 1)
        positions = torch.arange(self.num_points, device=x.device).unsqueeze(0)  # [1, N]
        pos_embed = self.position_embed(positions)
        seq += pos_embed
        transformed = self.transformer(seq)
        outputs = self.decoder(transformed)
        return outputs
        # return outputs.permute(0, 2, 1)


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # 参数编码器
        self.encoder = nn.Sequential(
            nn.Linear(9, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=3,
            bidirectional=False,
            batch_first=True
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # encode param [B,5] -> [B,256]
        encoded = self.encoder(x)

        # expand sequence [B,256] -> [B,256,256]
        repeated = encoded.unsqueeze(1).repeat(1, 256, 1)

        # LSTM [B,256,256] -> [B,256,512]
        lstm_out, _ = self.lstm(repeated)

        # decoded [B,256,512] -> [B,256,2]
        return self.decoder(lstm_out)


class CosmicNet2(nn.Module):
    def __init__(self):
        super().__init__()
        # fully connected network = CosmicNet II [N1]
        self.network = nn.Sequential(
            nn.Linear(9, 100),
            nn.LeakyReLU(negative_slope=0.25),  # Leaky ReLU, beta=0.25
            nn.LayerNorm(100),  # LN layer of SageNet
            nn.Linear(100, 250),
            nn.LeakyReLU(negative_slope=0.25),
            nn.LayerNorm(250),
            nn.Linear(250, 512)  # 256 points with 512 values (f_i, log10 OmegaGW)
        )

    def forward(self, x):
        # x: [B, 9] -> [B, 256*2]
        output = self.network(x)
        # [B, 256*2] -> [B, 256, 2]
        return output.view(-1, 256, 2)


class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(9, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )

        self.rnn = nn.RNN(
            input_size=256,
            hidden_size=256,
            num_layers=3,
            bidirectional=False,
            batch_first=True
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # encode param [B,5] -> [B,256]
        encoded = self.encoder(x)

        # expand sequence [B,256] -> [B,256,256]
        repeated = encoded.unsqueeze(1).repeat(1, 256, 1)

        # RNN [B,256,256] -> [B,256,256]
        rnn_out, _ = self.rnn(repeated)

        # decoded [B,256,256] -> [B,256,2]
        return self.decoder(rnn_out)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        return x + self.block(x)
