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


class LSTM(nn.Module):
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
