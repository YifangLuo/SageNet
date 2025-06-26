import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sagenetgw.models import LSTM, Former, CosmicNet2, RNN
from sagenetgw.classes import GWDataset
from tqdm import tqdm


def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    assert all(k in data[0] for k in ['r', 'n_t', 'kappa10', 'T_re', 'DN_re',
                                      'Omega_bh2', 'Omega_ch2', 'H0', 'A_s',
                                      'f_interp_85', 'log10OmegaGW_interp_85']), "Invalid data format."
    assert len(data[0]['f_interp_85']) == 256, "f_interp length should be 256"
    assert len(data[0]['log10OmegaGW_interp_85']) == 256, "log10OmegaGW length should be 256"
    return data


def collate_fn(batch):
    params, curves = zip(*batch)
    return torch.stack(params), torch.stack(curves)


def train_gw_model(json_path, model="Transformer", epochs=200, batch_size=32):
    raw_data = load_json_data(json_path)
    full_dataset = GWDataset(raw_data)
    print(f'JSON loaded. Total data num:{len(raw_data)}')

    train_idx, val_idx = train_test_split(
        np.arange(len(full_dataset)),
        test_size=0.2,
        random_state=42
    )
    train_data = torch.utils.data.Subset(full_dataset, train_idx)
    val_data = torch.utils.data.Subset(full_dataset, val_idx)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = model
    if model == 'LSTM':
        model = LSTM().to(device)
    elif model == 'Transformer':
        model = Former().to(device)
    elif model == 'CosmicNet2':
        model = CosmicNet2().to(device)
    elif model == 'RNN':
        model = RNN().to(device)
    else:
        raise ValueError(f'Unspecified model type "{model}".')

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    criterion = nn.MSELoss()
    print('Model initialized. Start training.')

    best_loss = float('inf')
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0

        for params, curves in train_loader:
            params = params.to(device)
            curves = curves.to(device)
            optimizer.zero_grad()
            outputs = model(params)
            loss = criterion(outputs, curves)
            # loss_last = criterion(outputs[:,-1, :], curves[:,-1,:]) * 5.0  # 权重设为5
            # loss_rest = criterion(outputs[:, :, :], curves[:, :, :])
            # loss = loss_last + loss_rest
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * params.size(0)

        # Valid
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for params, curves in val_loader:
                params = params.to(device)
                curves = curves.to(device)
                outputs = model(params)
                val_loss += criterion(outputs, curves).item() * params.size(0)

        # train_loss /= len(train_loader.dataset)
        # val_loss /= len(val_loader.dataset)
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4e} | Val Loss: {val_loss:.4e}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'model_state': model.state_dict(),
                'x_scaler': full_dataset.x_scaler,
                'y_scaler': full_dataset.y_scaler,
                'param_scaler': full_dataset.param_scaler
            }, f'best_gw_model_{model_name}_{len(raw_data)}.pth')

    return model


if __name__ == "__main__":
    trained_model = train_gw_model("solve_plus.data.json", model="LSTM", epochs=250)
