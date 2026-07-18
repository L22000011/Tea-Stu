import os
import random
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm


def generate_none_empth_modality_list():
    modality_list = random.choices(
        [True, False],
        k=3,
        weights=[50, 50],
    )
    lidar_ = random.choices(
        [True, False],
        k=1,
        weights=[70, 30],
    )
    wifi_ = random.choices(
        [True, False],
        k=1,
        weights=[70, 30],
    )
    final_list = modality_list + lidar_ + wifi_
    if sum(final_list) == 0:
        return generate_none_empth_modality_list()
    return final_list


def collate_fn_padd(batch):
    """
    dict_keys(['modality', 'scene', 'subject', 'action', 'idx', 'output',
    'input_vk', 'input_depth', 'input_lidar', 'input_mmwave'])
    """
    kpts = torch.FloatTensor(np.array([np.array(t['output']) for t in batch]))

    vk_data = torch.FloatTensor(np.array([t['input_vk'] for t in batch])).permute(0, 3, 1, 2)
    depth_data = torch.FloatTensor(np.array([t['input_depth'] for t in batch])).permute(0, 3, 1, 2)

    mmwave_data = [torch.Tensor(t['input_mmwave']) for t in batch]
    mmwave_data = torch.nn.utils.rnn.pad_sequence(mmwave_data).permute(1, 0, 2)

    lidar_data = [torch.Tensor(t['input_lidar'].copy()) for t in batch]
    lidar_data = torch.nn.utils.rnn.pad_sequence(lidar_data).permute(1, 0, 2)

    wifi_data = torch.FloatTensor(np.array([t['input_wifi-csi'] for t in batch]))
    modality_list = generate_none_empth_modality_list()

    return vk_data, depth_data, mmwave_data, lidar_data, wifi_data, kpts, modality_list


def _hpe_combo_specs():
    return [
        ('VK', [True, False, False, False, False]),
        ('Depth', [False, True, False, False, False]),
        ('Lidar', [False, False, False, True, False]),
        ('mmWave', [False, False, True, False, False]),
        ('WiFi-CSI', [False, False, False, False, True]),
        ('VK+Depth', [True, True, False, False, False]),
        ('VK+Lidar', [True, False, False, True, False]),
        ('VK+mmWave', [True, False, True, False, False]),
        ('VK+WiFi-CSI', [True, False, False, False, True]),
        ('Depth+Lidar', [False, True, False, True, False]),
        ('Depth+mmWave', [False, True, True, False, False]),
        ('Depth+WiFi-CSI', [False, True, False, False, True]),
        ('Lidar+mmWave', [False, False, True, True, False]),
        ('Lidar+WiFi-CSI', [False, False, False, True, True]),
        ('mmWave+WiFi-CSI', [False, False, True, False, True]),
        ('VK+Depth+Lidar', [True, True, False, True, False]),
        ('VK+Depth+mmWave', [True, True, True, False, False]),
        ('VK+Depth+WiFi-CSI', [True, True, False, False, True]),
        ('VK+Lidar+mmWave', [True, False, True, True, False]),
        ('VK+Lidar+WiFi-CSI', [True, False, False, True, True]),
        ('VK+mmWave+WiFi-CSI', [True, False, True, False, True]),
        ('Depth+Lidar+mmWave', [False, True, True, True, False]),
        ('Depth+Lidar+WiFi-CSI', [False, True, False, True, True]),
        ('Depth+mmWave+WiFi-CSI', [False, True, True, False, True]),
        ('Lidar+mmWave+WiFi-CSI', [False, False, True, True, True]),
        ('VK+Depth+Lidar+mmWave', [True, True, True, True, False]),
        ('VK+Depth+Lidar+WiFi-CSI', [True, True, False, True, True]),
        ('VK+Depth+mmWave+WiFi-CSI', [True, True, True, False, True]),
        ('VK+Lidar+mmWave+WiFi-CSI', [True, False, True, True, True]),
        ('Depth+Lidar+mmWave+WiFi-CSI', [False, True, True, True, True]),
        ('VK+Depth+Lidar+mmWave+WiFi-CSI', [True, True, True, True, True]),
    ]


def _move_hpe_batch_to_device(data, device):
    vk_data, depth_data, mmwave_data, lidar_data, wifi_data, kpts, modality_list = data
    vk_data = vk_data.to(device)
    depth_data = depth_data.to(device)
    mmwave_data = mmwave_data.to(device)
    lidar_data = lidar_data.to(device)
    wifi_data = wifi_data.to(device)
    labels = kpts.to(device).float()
    return vk_data, depth_data, mmwave_data, lidar_data, wifi_data, labels, modality_list


def _build_hpe_checkpoint(epoch, model, optimizer, best_test_mpjpe):
    return {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_test_mpjpe': best_test_mpjpe,
    }


def _load_model_weights(model, checkpoint):
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint
    model.load_state_dict(checkpoint)
    return None


def _remove_stale_checkpoint(path, label):
    if os.path.isfile(path):
        os.remove(path)
        print(f"[Ori-HPE] Removed stale {label}: {path}")


def _log_frozen_extractors(model):
    extractor = getattr(model, 'feature_extractor', None)
    if extractor is None:
        return
    names = [
        ('vk_extractor', 'VK'),
        ('depth_extractor', 'Depth'),
        ('mmwave_extractor', 'mmWave'),
        ('lidar_extractor', 'Lidar'),
        ('csi_extractor', 'WiFi-CSI'),
    ]
    for attr_name, display_name in names:
        module = getattr(extractor, attr_name, None)
        if module is None:
            continue
        is_frozen = all(not parameter.requires_grad for parameter in module.parameters())
        print(
            f"[Ori-HPE] Frozen extractor: {display_name} | "
            f"training={module.training} | requires_grad={is_frozen}"
        )


def hpe_test(model, tensor_loader, criterion1, criterion2, device, val_random_seed):
    model.eval()
    test_mpjpe = 0.0
    test_pampjpe = 0.0
    test_mse = 0.0
    random.seed(val_random_seed)
    with torch.no_grad():
        for data in tqdm(tensor_loader):
            vk_data, depth_data, mmwave_data, lidar_data, wifi_data, labels, modality_list = _move_hpe_batch_to_device(
                data, device
            )
            outputs = model(vk_data, depth_data, mmwave_data, lidar_data, wifi_data, modality_list).float()
            batch_size = vk_data.size(0)
            test_mse += criterion1(outputs, labels).item() * batch_size

            outputs_np = outputs.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            mpjpe, pampjpe = criterion2(outputs_np, labels_np)
            test_mpjpe += float(mpjpe) * batch_size
            test_pampjpe += float(pampjpe) * batch_size

    dataset_size = len(tensor_loader.dataset)
    test_mpjpe /= dataset_size
    test_pampjpe /= dataset_size
    test_mse /= dataset_size
    print("mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format(test_mse, test_mpjpe, test_pampjpe))
    return test_mpjpe


def hpe_train(
    model,
    train_loader,
    test_loader,
    num_epochs,
    learning_rate,
    train_criterion,
    test_criterion,
    device,
    save_dir,
    val_random_seed,
    resume_path=None,
    max_train_batches=1000,
):
    optimizer = torch.optim.AdamW(
        [
            {'params': model.linear_projector.parameters()},
            {'params': model.X_Fusion_block.parameters()},
        ],
        lr=learning_rate,
    )
    os.makedirs(save_dir, exist_ok=True)

    now_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    archive_parameter_path = os.path.join(save_dir, f'checkpoint_{now_time}.pth')
    last_parameter_path = os.path.join(save_dir, 'last.pth')
    best_parameter_path = os.path.join(save_dir, 'best.pth')

    best_test_mpjpe = float('inf')
    start_epoch = 0

    if resume_path and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        restored = _load_model_weights(model, checkpoint)
        if restored is not None:
            if 'optimizer_state_dict' in restored:
                optimizer.load_state_dict(restored['optimizer_state_dict'])
            start_epoch = restored.get('epoch', -1) + 1
            best_test_mpjpe = restored.get('best_test_mpjpe', best_test_mpjpe)
        print(f"[Ori-HPE] Resume training from epoch {start_epoch}.")
    else:
        print("[Ori-HPE] Start training from scratch.")
        _remove_stale_checkpoint(last_parameter_path, 'last checkpoint')
        _remove_stale_checkpoint(best_parameter_path, 'best checkpoint')

    for epoch in range(start_epoch, num_epochs):
        model.train()
        if epoch == start_epoch:
            _log_frozen_extractors(model)

        epoch_loss = 0.0
        epoch_samples = 0
        random.seed(epoch)
        num_iter = int(max_train_batches) if max_train_batches is not None else len(train_loader)

        for step, data in enumerate(tqdm(train_loader)):
            if step >= num_iter:
                break

            vk_data, depth_data, mmwave_data, lidar_data, wifi_data, labels, modality_list = _move_hpe_batch_to_device(
                data, device
            )
            optimizer.zero_grad()
            outputs = model(vk_data, depth_data, mmwave_data, lidar_data, wifi_data, modality_list).float()
            loss = train_criterion(outputs, labels)
            if torch.isnan(loss):
                raise RuntimeError('[Ori-HPE] Encountered NaN loss during training.')

            loss.backward()
            optimizer.step()

            batch_size = vk_data.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_samples += batch_size

        epoch_loss = epoch_loss / max(epoch_samples, 1)
        print(f'Epoch: {epoch}, Loss: {epoch_loss:.8f}')

        if (epoch + 1) % 10 == 0:
            test_mpjpe = hpe_test(
                model=model,
                tensor_loader=test_loader,
                criterion1=train_criterion,
                criterion2=test_criterion,
                device=device,
                val_random_seed=val_random_seed,
            )
            if test_mpjpe <= best_test_mpjpe:
                best_test_mpjpe = test_mpjpe
                torch.save(
                    _build_hpe_checkpoint(epoch, model, optimizer, best_test_mpjpe),
                    best_parameter_path,
                )
                print(f"[Ori-HPE] Updated best checkpoint with mpjpe={test_mpjpe:.8f}")

        torch.save(
            _build_hpe_checkpoint(epoch, model, optimizer, best_test_mpjpe),
            last_parameter_path,
        )
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            epoch_parameter_path = os.path.join(save_dir, f'epoch_{epoch + 1:03d}.pth')
            torch.save(
                _build_hpe_checkpoint(epoch, model, optimizer, best_test_mpjpe),
                epoch_parameter_path,
            )

    final_checkpoint = _build_hpe_checkpoint(num_epochs - 1, model, optimizer, best_test_mpjpe)
    torch.save(final_checkpoint, archive_parameter_path)
    if not os.path.isfile(best_parameter_path):
        torch.save(final_checkpoint, best_parameter_path)
        print("[Ori-HPE] Best checkpoint was absent; saved final checkpoint as best.pth")
    return


def multi_test(model, tensor_loader, criterion1, criterion2, device, val_random_seed, max_batches=None):
    model.eval()
    combo_specs = _hpe_combo_specs()
    totals = {
        name: {'mse': 0.0, 'mpjpe': 0.0, 'pampjpe': 0.0}
        for name, _ in combo_specs
    }

    random.seed(val_random_seed)
    with torch.no_grad():
        seen_samples = 0
        for batch_idx, data in enumerate(tqdm(tensor_loader)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            vk_data, depth_data, mmwave_data, lidar_data, wifi_data, labels, _ = _move_hpe_batch_to_device(data, device)
            batch_size = vk_data.size(0)
            seen_samples += batch_size
            labels_np = labels.detach().cpu().numpy()

            for name, modality_list in combo_specs:
                outputs = model(vk_data, depth_data, mmwave_data, lidar_data, wifi_data, modality_list).float()
                totals[name]['mse'] += criterion1(outputs, labels).item() * batch_size

                outputs_np = outputs.detach().cpu().numpy()
                mpjpe, pampjpe = criterion2(outputs_np, labels_np)
                totals[name]['mpjpe'] += float(mpjpe) * batch_size
                totals[name]['pampjpe'] += float(pampjpe) * batch_size

    dataset_size = seen_samples if max_batches is not None else len(tensor_loader.dataset)
    for name, _ in combo_specs:
        mse = totals[name]['mse'] / dataset_size
        mpjpe = totals[name]['mpjpe'] / dataset_size
        pampjpe = totals[name]['pampjpe'] / dataset_size
        print(
            "modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format(
                name,
                mse,
                mpjpe,
                pampjpe,
            )
        )
