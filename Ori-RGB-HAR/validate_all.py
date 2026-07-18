import csv
import io
import os
import re
import sys
from contextlib import redirect_stdout
from datetime import datetime

import numpy as np
import torch
from torch import nn
import yaml
from syn_DI_dataset import make_dataset, make_dataloader
from utils import collate_fn_padd, multi_test
from X_Fi import X_Fi
import argparse


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def save_eval_outputs(text, csv_path):
    pattern = re.compile(
        r"modality:\s*(.*?),\s*Cross Entropy Loss:\s*([0-9eE+\-.]+),\s*Accuracy:\s*([0-9eE+\-.]+)"
    )
    rows = []
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            rows.append(
                {
                    'modality': match.group(1).strip(),
                    'cross_entropy_loss': match.group(2),
                    'accuracy': match.group(3),
                }
            )
    with open(csv_path, 'w', newline='', encoding='utf-8') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=['modality', 'cross_entropy_loss', 'accuracy'])
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        return 'checkpoint_dict'
    model.load_state_dict(checkpoint)
    return 'state_dict'


def main():
    parser = argparse.ArgumentParser('X-Fi model for MMFI HAR')
    parser.add_argument('--dataset', type=str, required=True, help='path to dataset, e.g. d:/Data/My_MMFi_Data/MMFi_Dataset')
    parser.add_argument('--pt_weights', type=str, required=True, help='path to pretrained model weights, e.g. ./pre-trained_weights/mmfi_har_checkpoint.pt')
    parser.add_argument('--config', type=str, default='config.yaml', help='path to config yaml')
    parser.add_argument('--outputs-dir', type=str, default='./outputs', help='directory for validation text/csv outputs')
    parser.add_argument('--max-eval-batches', type=int, default=None, help='limit evaluation batches for smoke tests')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Available validation resources:{device}')

    with open(args.config, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    train_dataset, val_dataset = make_dataset(args.dataset, config)
    rng_generator = torch.manual_seed(config['init_rand_seed'])
    val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['val_loader'], collate_fn=collate_fn_padd)

    torch.manual_seed(3407)
    model = X_Fi(model_depth=2)
    model.to(device)
    checkpoint_kind = load_checkpoint(model, args.pt_weights, device)
    print(f'[Ori-HAR] Use checkpoint: {args.pt_weights}')
    print(f'[Ori-HAR] Checkpoint format: {checkpoint_kind}')

    criterion = nn.CrossEntropyLoss()
    val_random_seed = config['modality_existances']['val_random_seed']

    outputs_dir = args.outputs_dir
    os.makedirs(outputs_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    txt_path = os.path.join(outputs_dir, f'main_eval_{timestamp}.txt')
    csv_path = os.path.join(outputs_dir, f'main_eval_{timestamp}.csv')
    latest_txt_path = os.path.join(outputs_dir, 'main_eval_latest.txt')
    latest_csv_path = os.path.join(outputs_dir, 'main_eval_latest.csv')

    capture_buffer = io.StringIO()
    with open(txt_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f'dataset: {args.dataset}\n')
        log_file.write(f'checkpoint: {args.pt_weights}\n')
        log_file.write(f'timestamp: {timestamp}\n\n')
        with redirect_stdout(Tee(sys.stdout, log_file, capture_buffer)):
            multi_test(model, val_loader, criterion, device, val_random_seed, max_batches=args.max_eval_batches)

    text = capture_buffer.getvalue()
    with open(latest_txt_path, 'w', encoding='utf-8') as latest_file:
        latest_file.write(text)

    save_eval_outputs(text, csv_path)
    save_eval_outputs(text, latest_csv_path)

    print(f'[Ori-HAR] Saved evaluation text: {txt_path}')
    print(f'[Ori-HAR] Saved evaluation csv: {csv_path}')
    return


if __name__ == '__main__':
    main()
