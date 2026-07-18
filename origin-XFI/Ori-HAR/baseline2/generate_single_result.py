import argparse
import os

import numpy as np
import torch
import yaml
from tqdm import tqdm

from baseline_model import single_model
from syn_DI_dataset import make_dataloader, make_dataset


def collate_fn_padd(batch):
    '''
    Padds batch of variable length.

    dict_keys(['modality', 'scene', 'subject', 'action', 'idx', 'output',
    'input_vk', 'input_depth', 'input_lidar', 'input_mmwave'])
    '''
    all_actions = {'A01': 0., 'A02': 1., 'A03': 2., 'A04': 3., 'A05': 4.,
                   'A06': 5., 'A07': 6., 'A08': 7., 'A09': 8., 'A10': 9.,
                   'A11': 10., 'A12': 11., 'A13': 12., 'A14': 13., 'A15': 14.,
                   'A16': 15., 'A17': 16., 'A18': 17., 'A19': 18., 'A20': 19.,
                   'A21': 20., 'A22': 21., 'A23': 22., 'A24': 23., 'A25': 24.,
                   'A26': 25., 'A27': 26.}
    labels = []
    [labels.append(all_actions[t['action']]) for t in batch]
    labels = torch.FloatTensor(labels)

    vk_data = np.array([(t['input_vk']) for t in batch])
    vk_data = torch.FloatTensor(vk_data)

    depth_data = np.array([(t['input_depth']) for t in batch])
    depth_data = torch.FloatTensor(depth_data).permute(0, 3, 1, 2)

    mmwave_data = [torch.Tensor(t['input_mmwave']) for t in batch]
    mmwave_data = torch.nn.utils.rnn.pad_sequence(mmwave_data)
    mmwave_data = mmwave_data.permute(1, 0, 2)

    lidar_data = [torch.Tensor(t['input_lidar']) for t in batch]
    lidar_data = torch.nn.utils.rnn.pad_sequence(lidar_data)
    lidar_data = lidar_data.permute(1, 0, 2)

    exist_list = [True, False, False, False]
    return vk_data, depth_data, lidar_data, mmwave_data, labels, exist_list


def get_result(vk_model, depth_model, mmwave_model, lidar_model, tensor_loader, device, result_dir):
    vk_model.eval()
    depth_model.eval()
    mmwave_model.eval()
    lidar_model.eval()
    os.makedirs(result_dir, exist_ok=True)

    for i, data in tqdm(enumerate(tensor_loader)):
        vk_data, depth_data, lidar_data, mmwave_data, label, exist_list = data
        vk_data = vk_data.to(device)
        depth_data = depth_data.to(device)
        lidar_data = lidar_data.to(device)
        mmwave_data = mmwave_data.to(device)
        label.to(device)
        labels = label.type(torch.FloatTensor)

        vk_outputs = vk_model(vk_data, [True, False, False, False])
        depth_outputs = depth_model(depth_data, [False, True, False, False])
        mmwave_outputs = mmwave_model(mmwave_data, [False, False, True, False])
        lidar_outputs = lidar_model(lidar_data, [False, False, False, True])

        vk_outputs = vk_outputs.detach().cpu().numpy()
        depth_outputs = depth_outputs.detach().cpu().numpy()
        mmwave_outputs = mmwave_outputs.detach().cpu().numpy()
        lidar_outputs = lidar_outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        if i == 0:
            vk_result = vk_outputs
            depth_result = depth_outputs
            mmwave_result = mmwave_outputs
            lidar_result = lidar_outputs
            all_label = labels
        else:
            vk_result = np.vstack((vk_result, vk_outputs))
            depth_result = np.vstack((depth_result, depth_outputs))
            mmwave_result = np.vstack((mmwave_result, mmwave_outputs))
            lidar_result = np.vstack((lidar_result, lidar_outputs))
            all_label = np.hstack((all_label, labels))

    np.save(os.path.join(result_dir, 'vk_result.npy'), vk_result)
    np.save(os.path.join(result_dir, 'depth_result.npy'), depth_result)
    np.save(os.path.join(result_dir, 'mmwave_result.npy'), mmwave_result)
    np.save(os.path.join(result_dir, 'lidar_result.npy'), lidar_result)
    np.save(os.path.join(result_dir, 'all_label.npy'), all_label)


def load_single_model(modality, ckpt_path, device):
    model = single_model([modality])
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser('Official-style X-Fi HAR baseline2 with VK')
    parser.add_argument('--dataset', type=str, required=True, help='MMFi dataset root')
    parser.add_argument('--device', type=str, default='cuda:0', help='device, e.g. cuda:0')
    parser.add_argument('--weights-dir', type=str, default='../baseline1/baseline_weights', help='directory containing baseline1 single-modality weights')
    parser.add_argument('--results-dir', type=str, default='./baseline_results', help='directory to save generated baseline2 results')
    args = parser.parse_args()

    with open('config.yaml', 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    train_dataset, val_dataset = make_dataset(args.dataset, config)
    rng_generator = torch.manual_seed(config['init_rand_seed'])
    train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['train_loader'], collate_fn=collate_fn_padd)
    val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['val_loader'], collate_fn=collate_fn_padd)
    _ = train_loader

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    vk_model = load_single_model('vk', os.path.join(args.weights_dir, 'vk_.pt'), device)
    depth_model = load_single_model('depth', os.path.join(args.weights_dir, 'depth_.pt'), device)
    mmwave_model = load_single_model('mmwave', os.path.join(args.weights_dir, 'mmwave_.pt'), device)
    lidar_model = load_single_model('lidar', os.path.join(args.weights_dir, 'lidar_.pt'), device)

    get_result(vk_model, depth_model, mmwave_model, lidar_model, val_loader, device, args.results_dir)


if __name__ == '__main__':
    main()
