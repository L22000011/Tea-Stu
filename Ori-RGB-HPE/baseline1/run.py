import argparse
import os

import numpy as np
import torch
import yaml
from torch import nn
from tqdm import tqdm

from baseline_model import Five_model, dual_model, quadra_model, single_model, triple_model
from evaluate import error
from syn_DI_dataset import make_dataloader, make_dataset


def collate_fn_padd(batch):
    '''
    Padds batch of variable length.

    dict_keys(['modality', 'scene', 'subject', 'action', 'idx', 'output',
    'input_vk', 'input_depth', 'input_lidar', 'input_mmwave'])
    '''
    for t in batch:
        dict_keys = t.keys()

    kpts = []
    [kpts.append(np.array(t['output'])) for t in batch]
    kpts = torch.FloatTensor(np.array(kpts))

    return_data = []
    exist_list = []

    if 'input_vk' in dict_keys:
        vk_data = np.array([(t['input_vk']) for t in batch])
        vk_data = torch.FloatTensor(vk_data).permute(0, 3, 1, 2)
        return_data.append(vk_data)
        exist_list.append(True)
    else:
        exist_list.append(False)

    if 'input_depth' in dict_keys:
        depth_data = np.array([(t['input_depth']) for t in batch])
        depth_data = torch.FloatTensor(depth_data).permute(0, 3, 1, 2)
        return_data.append(depth_data)
        exist_list.append(True)
    else:
        exist_list.append(False)

    if 'input_mmwave' in dict_keys:
        mmwave_data = [torch.Tensor(t['input_mmwave']) for t in batch]
        mmwave_data = torch.nn.utils.rnn.pad_sequence(mmwave_data)
        mmwave_data = mmwave_data.permute(1, 0, 2)
        return_data.append(mmwave_data)
        exist_list.append(True)
    else:
        exist_list.append(False)

    if 'input_lidar' in dict_keys:
        lidar_data = [torch.Tensor(t['input_lidar'].copy()) for t in batch]
        lidar_data = torch.nn.utils.rnn.pad_sequence(lidar_data)
        lidar_data = lidar_data.permute(1, 0, 2)
        return_data.append(lidar_data)
        exist_list.append(True)
    else:
        exist_list.append(False)

    if 'input_wifi-csi' in dict_keys:
        wifi_data = np.array([(t['input_wifi-csi']) for t in batch])
        wifi_data = torch.FloatTensor(wifi_data)
        return_data.append(wifi_data)
        exist_list.append(True)
    else:
        exist_list.append(False)

    return return_data, kpts, exist_list


def test(model, tensor_loader, criterion1, criterion2, device):
    model.eval()
    test_mpjpe = 0
    test_pampjpe = 0
    test_mse = 0
    for data in tqdm(tensor_loader):
        input_data, kpts, exist_list = data
        for i, modal_data in enumerate(input_data):
            modal_data = modal_data.to(device)
            globals()[f'input_{str(i + 1)}'] = modal_data
        labels = kpts.to(device).float()
        if len(input_data) == 1:
            outputs = model(input_1, exist_list)
        elif len(input_data) == 2:
            outputs = model(input_1, input_2, exist_list)
        elif len(input_data) == 3:
            outputs = model(input_1, input_2, input_3, exist_list)
        elif len(input_data) == 4:
            outputs = model(input_1, input_2, input_3, input_4, exist_list)
        elif len(input_data) == 5:
            outputs = model(input_1, input_2, input_3, input_4, input_5, exist_list)
        else:
            raise ValueError('error in input_data')
        outputs = outputs.float()
        test_mse += criterion1(outputs, labels).item() * input_1.size(0)

        outputs = outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        mpjpe, pampjpe = criterion2(outputs, labels)
        test_mpjpe += mpjpe.item() * input_1.size(0)
        test_pampjpe += pampjpe.item() * input_1.size(0)
    test_mpjpe = test_mpjpe / len(tensor_loader.dataset)
    test_pampjpe = test_pampjpe / len(tensor_loader.dataset)
    test_mse = test_mse / len(tensor_loader.dataset)
    print("mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format(float(test_mse), float(test_mpjpe), float(test_pampjpe)))
    return test_mpjpe


def train(model, train_loader, test_loader, num_epochs, learning_rate, train_criterion, test_criterion, device, modality, save_dir):
    optim_groups = [{'params': model.regression_head.parameters()}]
    if hasattr(model.feature_extractor, 'rgb_extractor'):
        optim_groups.insert(0, {'params': model.feature_extractor.rgb_extractor.parameters()})
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)

    os.makedirs(save_dir, exist_ok=True)
    name = ''
    for mod in modality:
        name = name + mod + '_'
    name = name + '.pt'
    parameter_dir = os.path.join(save_dir, name)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_iter = 400
        for i, data in enumerate(tqdm(train_loader)):
            if i < num_iter:
                input_data, kpts, exist_list = data
                for i, modal_data in enumerate(input_data):
                    modal_data = modal_data.to(device)
                    globals()[f'input_{str(i + 1)}'] = modal_data
                labels = kpts.to(device).float()
                optimizer.zero_grad()
                if len(input_data) == 1:
                    outputs = model(input_1, exist_list)
                elif len(input_data) == 2:
                    outputs = model(input_1, input_2, exist_list)
                elif len(input_data) == 3:
                    outputs = model(input_1, input_2, input_3, exist_list)
                elif len(input_data) == 4:
                    outputs = model(input_1, input_2, input_3, input_4, exist_list)
                elif len(input_data) == 5:
                    outputs = model(input_1, input_2, input_3, input_4, input_5, exist_list)
                else:
                    raise ValueError('error in exist_list')
                outputs = outputs.float()
                loss = train_criterion(outputs, labels)
                if loss == float('nan'):
                    print('nan')
                    print(outputs)
                    print(labels)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * input_1.size(0)
            else:
                break
        epoch_loss = epoch_loss / (input_1.size(0) * num_iter)
        print('Epoch: {}, Loss: {:.8f}'.format(epoch, epoch_loss))
        if (epoch + 1) % 5 == 0:
            test_mpjpe = test(
                model=model,
                tensor_loader=test_loader,
                criterion1=train_criterion,
                criterion2=test_criterion,
                device=device
            )
            print(f"test mpjpe is:{test_mpjpe}")
    torch.save(model.state_dict(), parameter_dir)


def build_model(modality_list):
    if len(modality_list) == 1:
        return single_model(modality_list)
    if len(modality_list) == 2:
        return dual_model(modality_list)
    if len(modality_list) == 3:
        return triple_model(modality_list)
    if len(modality_list) == 4:
        return quadra_model(modality_list)
    if len(modality_list) == 5:
        return Five_model(modality_list)
    raise ValueError('Unsupported modality list length')


def main():
    parser = argparse.ArgumentParser('Official-style X-Fi HPE baseline1 with VK')
    parser.add_argument('--dataset', type=str, required=True, help='MMFi dataset root')
    parser.add_argument('--device', type=str, default='cuda:0', help='device, e.g. cuda:0')
    parser.add_argument('--weights-dir', type=str, default='./baseline_weights', help='directory to save trained baseline weights')
    args = parser.parse_args()

    with open('config_all.yaml', 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    for i in range(len(config['modality_list'])):
        config['modality'] = config['modality_list'][i]
        train_dataset, val_dataset = make_dataset(args.dataset, config)
        rng_generator = torch.manual_seed(config['init_rand_seed'])
        train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['loader'], collate_fn=collate_fn_padd)
        val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['loader'], collate_fn=collate_fn_padd)

        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(device)

        model = build_model(config['modality'])
        model.to(device)

        train_criterion = nn.MSELoss()
        test_criterion = error

        test(
            model=model,
            tensor_loader=val_loader,
            criterion1=train_criterion,
            criterion2=test_criterion,
            device=device
        )
        train(
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            num_epochs=35,
            learning_rate=1e-3,
            train_criterion=train_criterion,
            test_criterion=test_criterion,
            device=device,
            modality=config['modality'],
            save_dir=args.weights_dir
        )


if __name__ == '__main__':
    main()
