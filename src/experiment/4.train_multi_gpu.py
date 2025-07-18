import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist  # [CHANGE] Import distributed module
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler  # [CHANGE] Import DistributedSampler
from sklearn.model_selection import train_test_split
from sagenetgw.models import LSTM, Former, CosmicNet2, RNN, CNN, TCN
from sagenetgw.classes import GWDataset
from tqdm import tqdm


def load_json_data(file_path, f_name, omega_name):
    with open(file_path, 'r') as f:
        data = json.load(f)
    assert all(k in data[0] for k in ['r', 'n_t', 'kappa10', 'T_re', 'DN_re',
                                      'Omega_bh2', 'Omega_ch2', 'H0', 'A_s',
                                      f_name, omega_name]), "Invalid data format."
    assert len(data[0][f_name]) == 256, "f_interp length should be 256"
    assert len(data[0][omega_name]) == 256, "log10OmegaGW_interp length should be 256"
    return data


def collate_fn(batch):
    params, curves = zip(*batch)
    return torch.stack(params), torch.stack(curves)


def train_gw_model(json_path, model="Transformer", epochs=200, batch_size=32,
                   interp_percent=None):
    # [CHANGE] Initialize distributed training
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])  # Get local rank for GPU assignment
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    raw_data = load_json_data(json_path, f_name=f'f_interp_{interp_percent}',
                              omega_name=f'log10OmegaGW_interp_{interp_percent}')
    full_dataset = GWDataset(raw_data, interp_percent=interp_percent)

    # [CHANGE] Only print on rank 0 to avoid duplicate output
    if rank == 0:
        print(f'JSON loaded. Total data num:{len(raw_data)}. model:{model}; percent:{interp_percent}')

    train_idx, val_idx = train_test_split(
        np.arange(len(full_dataset)),
        test_size=0.2,
        random_state=42
    )
    train_data = torch.utils.data.Subset(full_dataset, train_idx)
    val_data = torch.utils.data.Subset(full_dataset, val_idx)

    # [CHANGE] Use DistributedSampler for training data
    train_sampler = DistributedSampler(train_data, shuffle=True)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,  # [CHANGE] Use sampler instead of shuffle=True
        collate_fn=collate_fn,
        num_workers=4,  # [CHANGE] Add num_workers for faster data loading
        pin_memory=True  # [CHANGE] Enable pin_memory for faster data transfer to GPU
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=4,  # [CHANGE] Add num_workers
        pin_memory=True  # [CHANGE] Enable pin_memory
    )

    model_name = model
    if model == 'LSTM':
        model = LSTM().to(device)
    elif model == 'Transformer':
        model = Former().to(device)
    elif model == 'CosmicNet2':
        model = CosmicNet2().to(device)
    elif model == 'RNN':
        model = RNN().to(device)
    elif model == 'CNN':
        model = CNN().to(device)
    elif model == 'TCN':
        model = TCN().to(device)
    else:
        raise ValueError(f'Unspecified model type "{model}".')

    # [CHANGE] Wrap model with DDP
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    criterion = nn.MSELoss()

    # [CHANGE] Only print on rank 0
    if rank == 0:
        print(f'Model initialized. Start training. Current device:{device}')

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_sampler.set_epoch(epoch)  # [CHANGE] Ensure different data shuffling each epoch

        for params, curves in train_loader:
            params = params.to(device)
            curves = curves.to(device)
            optimizer.zero_grad()
            outputs = model(params)
            loss = criterion(outputs, curves)
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

        # [CHANGE] Fix loss normalization: divide by dataset size, not number of batches
        train_loss /= len(train_data)
        val_loss /= len(val_data)

        # [CHANGE] Aggregate losses across GPUs
        train_loss_tensor = torch.tensor(train_loss).to(device)
        val_loss_tensor = torch.tensor(val_loss).to(device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        train_loss = train_loss_tensor.item() / dist.get_world_size()
        val_loss = val_loss_tensor.item() / dist.get_world_size()

        scheduler.step(val_loss)

        # [CHANGE] Only print and save on rank 0
        if rank == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4e} | Val Loss: {val_loss:.4e}")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'model_state': model.state_dict(),
                    'x_scaler': full_dataset.x_scaler,
                    'y_scaler': full_dataset.y_scaler,

                    'param_scaler': full_dataset.param_scaler
                }, f'best_gw_model_{model_name}_{interp_percent}_{len(raw_data)}.pth')

    # [CHANGE] Clean up distributed process group
    dist.destroy_process_group()

    return model


if __name__ == "__main__":
    # [CHANGE] Import os for environment variables
    import os

    parser = argparse.ArgumentParser(description='Train the model by specific dataset.')
    parser.add_argument('--percent', type=int, required=True,
                        help='interp percent')
    parser.add_argument('--model', type=str, required=True,
                        help='model')
    args = parser.parse_args()

    percent = args.percent
    model = args.model

    trained_model = train_gw_model(f"./full.interp.{percent}.json", model=model, epochs=250,
                                   interp_percent=percent)