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
        # encode param [B,9] -> [B,256]
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
        # encode param [B,9] -> [B,256]
        encoded = self.encoder(x)

        # expand sequence [B,256] -> [B,256,256]
        repeated = encoded.unsqueeze(1).repeat(1, 256, 1)

        # RNN [B,256,256] -> [B,256,256]
        rnn_out, _ = self.rnn(repeated)

        # decoded [B,256,256] -> [B,256,2]
        return self.decoder(rnn_out)


class CNN(nn.Module):
    def __init__(self, num_points=256):
        super().__init__()
        self.num_points = num_points

        # Parameter encoder: [B, 9] -> [B, 256]
        self.encoder = nn.Sequential(
            nn.Linear(9, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )

        # CNN layers: process sequence data
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.LayerNorm([256, num_points]),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.LayerNorm([256, num_points])
        )

        # Decoder: [B, 256, 256] -> [B, 256, 2]
        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # Encode input: [B, 9] -> [B, 256]
        encoded = self.encoder(x)

        # Expand to sequence: [B, 256] -> [B, 256, 256]
        seq = encoded.unsqueeze(1).repeat(1, self.num_points, 1)

        # Transpose for Conv1d: [B, 256, 256] -> [B, 256, 256]
        seq = seq.transpose(1, 2)

        # Apply CNN: [B, 256, 256] -> [B, 256, 256]
        conv_out = self.conv_layers(seq)

        # Transpose back: [B, 256, 256] -> [B, 256, 256]
        conv_out = conv_out.transpose(1, 2)

        # Decode: [B, 256, 256] -> [B, 256, 2]
        output = self.decoder(conv_out)

        return output

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        # Causal padding: (kernel_size - 1) * dilation
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            padding_mode='zeros'
        )
        self.norm = nn.LayerNorm([out_channels, 256])
        self.gelu = nn.GELU()
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # Causal conv: trim padding from right to ensure causality
        out = self.conv(x)[:, :, :-(self.conv.padding[0])]
        out = self.gelu(out)
        out = self.norm(out)
        # Add residual connection
        return out + self.residual(x)

class TCN(nn.Module):
    def __init__(self, num_points=256):
        super().__init__()
        self.num_points = num_points

        # Parameter encoder: [B, 9] -> [B, 256]
        self.encoder = nn.Sequential(
            nn.Linear(9, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )

        # TCN layers with increasing dilations
        self.tcn_layers = nn.Sequential(
            TCNBlock(in_channels=256, out_channels=256, kernel_size=3, dilation=1),
            TCNBlock(in_channels=256, out_channels=256, kernel_size=3, dilation=2),
            TCNBlock(in_channels=256, out_channels=256, kernel_size=3, dilation=4)
        )

        # Decoder: [B, 256, 256] -> [B, 256, 2]
        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # Encode input: [B, 9] -> [B, 256]
        encoded = self.encoder(x)

        # Expand to sequence: [B, 256] -> [B, 256, 256]
        seq = encoded.unsqueeze(1).repeat(1, self.num_points, 1)

        # Transpose for Conv1d: [B, 256, 256] -> [B, 256, 256]
        seq = seq.transpose(1, 2)

        # Apply TCN: [B, 256, 256] -> [B, 256, 256]
        tcn_out = self.tcn_layers(seq)

        # Transpose back: [B, 256, 256] -> [B, 256, 256]
        tcn_out = tcn_out.transpose(1, 2)

        # Decode: [B, 256, 256] -> [B, 256, 2]
        output = self.decoder(tcn_out)

        return output


class GRU(nn.Module):
    def __init__(self, num_points=256):
        super().__init__()
        self.num_points = num_points

        # Parameter encoder: matches Former and LSTM
        self.encoder = nn.Sequential(
            nn.Linear(9, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )

        # GRU: 3 layers to match LSTM and Transformer
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=256,
            num_layers=3,
            batch_first=True,
            bidirectional=False
        )

        # Decoder: matches Former and LSTM
        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # x: [batch, 9]
        # Encode parameters: [batch, 9] -> [batch, 256]
        encoded = self.encoder(x)

        # Expand to sequence: [batch, 256] -> [batch, num_points, 256]
        seq = encoded.unsqueeze(1).repeat(1, self.num_points, 1)

        # GRU processing: [batch, num_points, 256] -> [batch, num_points, 256]
        gru_out, _ = self.gru(seq)

        # Decode: [batch, num_points, 256] -> [batch, num_points, 2]
        outputs = self.decoder(gru_out)

        return outputs