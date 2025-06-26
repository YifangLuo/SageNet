import torch
from collections import OrderedDict


def fix_pth_file(input_path, output_path):
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state']

    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('module.'):
            # remove "module."
            # e.g. module.encoder.embedding.weight -> encoder.embedding.weight
            new_key = key[len('module.'):]
            new_state_dict[new_key] = value
        else:
            # if no "module." then directly save
            new_state_dict[key] = value

    # update checkpoint and save
    checkpoint['model_state'] = new_state_dict
    torch.save(checkpoint, output_path)
    print(f"Successfully removed 'module.' in {input_path} and saved to {output_path}")


if __name__ == "__main__":
    fix_pth_file('best_gw_model_LSTM_60_185578.pth', 'best_gw_model_LSTM_60_185578.pth')
