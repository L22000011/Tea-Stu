import os
import random
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm


def generate_none_empth_modality_list():
    vk_ = random.choices(
        [True, False],
        k=1,
        weights=[50, 50],
    )
    depth_ = random.choices(
        [True, False],
        k=1,
        weights=[60, 40],
    )
    mmwave_ = random.choices(
        [True, False],
        k=1,
        weights=[50, 50],
    )
    lidar_ = random.choices(
        [True, False],
        k=1,
        weights=[90, 10],
    )
    modality_list = vk_ + depth_ + mmwave_ + lidar_
    if sum(modality_list) == 0:
        return generate_none_empth_modality_list()
    return modality_list


def collate_fn_padd(batch):
    """
    dict_keys(['modality', 'scene', 'subject', 'action', 'idx', 'output',
    'input_vk', 'input_depth', 'input_lidar', 'input_mmwave'])
    """
    all_actions = {
        'A01': 0.0, 'A02': 1.0, 'A03': 2.0, 'A04': 3.0, 'A05': 4.0,
        'A06': 5.0, 'A07': 6.0, 'A08': 7.0, 'A09': 8.0, 'A10': 9.0,
        'A11': 10.0, 'A12': 11.0, 'A13': 12.0, 'A14': 13.0, 'A15': 14.0,
        'A16': 15.0, 'A17': 16.0, 'A18': 17.0, 'A19': 18.0, 'A20': 19.0,
        'A21': 20.0, 'A22': 21.0, 'A23': 22.0, 'A24': 23.0, 'A25': 24.0,
        'A26': 25.0, 'A27': 26.0,
    }
    labels = torch.FloatTensor([all_actions[t['action']] for t in batch])

    vk_array = np.array([t['input_vk'] for t in batch])
    vk_data = torch.FloatTensor(vk_array)
    if vk_data.ndim == 4 and vk_data.shape[-1] in [1, 3]:
        vk_data = vk_data.permute(0, 3, 1, 2)
    depth_data = torch.FloatTensor(np.array([t['input_depth'] for t in batch])).permute(0, 3, 1, 2)

    mmwave_data = [torch.Tensor(t['input_mmwave']) for t in batch]
    mmwave_data = torch.nn.utils.rnn.pad_sequence(mmwave_data).permute(1, 0, 2)

    lidar_data = [torch.Tensor(t['input_lidar'].copy()) for t in batch]
    lidar_data = torch.nn.utils.rnn.pad_sequence(lidar_data).permute(1, 0, 2)

    modality_list = generate_none_empth_modality_list()
    return vk_data, depth_data, mmwave_data, lidar_data, labels, modality_list


def _har_combo_specs():
    return [
        ('VK', [True, False, False, False]),
        ('Depth', [False, True, False, False]),
        ('Lidar', [False, False, False, True]),
        ('mmWave', [False, False, True, False]),
        ('VK+Depth', [True, True, False, False]),
        ('VK+Lidar', [True, False, False, True]),
        ('VK+mmWave', [True, False, True, False]),
        ('Depth+Lidar', [False, True, False, True]),
        ('Depth+mmWave', [False, True, True, False]),
        ('Lidar+mmWave', [False, False, True, True]),
        ('VK+Depth+Lidar', [True, True, False, True]),
        ('VK+Depth+mmWave', [True, True, True, False]),
        ('VK+Lidar+mmWave', [True, False, True, True]),
        ('Depth+Lidar+mmWave', [False, True, True, True]),
        ('VK+Depth+Lidar+mmWave', [True, True, True, True]),
    ]


def _move_har_batch_to_device(data, device):
    vk_data, depth_data, mmwave_data, lidar_data, label, modality_list = data
    vk_data = vk_data.to(device)
    depth_data = depth_data.to(device)
    mmwave_data = mmwave_data.to(device)
    lidar_data = lidar_data.to(device)
    labels = label.to(device).long()
    return vk_data, depth_data, mmwave_data, lidar_data, labels, modality_list


def _build_har_checkpoint(epoch, model, optimizer, scheduler, best_test_acc):
    return {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_test_acc': best_test_acc,
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
        print(f"[Ori-HAR] Removed stale {label}: {path}")


def har_test(model, tensor_loader, criterion, device, val_random_seed):
    model.eval()
    test_correct = 0
    test_loss = 0.0
    random.seed(val_random_seed)
    with torch.no_grad():
        for data in tqdm(tensor_loader):
            vk_data, depth_data, mmwave_data, lidar_data, labels, modality_list = _move_har_batch_to_device(data, device)
            outputs = model(vk_data, depth_data, mmwave_data, lidar_data, modality_list).float()
            batch_size = labels.size(0)
            loss = criterion(outputs, labels)
            predict_y = torch.argmax(outputs, dim=1)
            test_correct += (predict_y == labels).sum().item()
            test_loss += loss.item() * batch_size

    dataset_size = len(tensor_loader.dataset)
    test_acc = test_correct / dataset_size
    test_loss = test_loss / dataset_size
    print("validation accuracy:{:.4f}, loss:{:.5f}".format(test_acc, test_loss))
    return test_acc


def har_train(
    model,
    train_loader,
    test_loader,
    num_epochs,
    learning_rate,
    criterion,
    device,
    save_dir,
    val_random_seed,
    resume_path=None,
    max_train_batches=1000,
):
    optimizer = torch.optim.AdamW(
        [
            {'params': model.feature_extractor.vk_extractor.parameters()},
            {'params': model.linear_projector.parameters()},
            {'params': model.X_Fusion_block.parameters()},
        ],
        lr=learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)
    os.makedirs(save_dir, exist_ok=True)

    now_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    archive_parameter_path = os.path.join(save_dir, f'checkpoint_{now_time}.pth')
    last_parameter_path = os.path.join(save_dir, 'last.pth')
    best_parameter_path = os.path.join(save_dir, 'best.pth')

    best_test_acc = 0.0
    start_epoch = 0

    if resume_path and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        restored = _load_model_weights(model, checkpoint)
        if restored is not None:
            if 'optimizer_state_dict' in restored:
                optimizer.load_state_dict(restored['optimizer_state_dict'])
            if 'scheduler_state_dict' in restored:
                scheduler.load_state_dict(restored['scheduler_state_dict'])
            start_epoch = restored.get('epoch', -1) + 1
            best_test_acc = restored.get('best_test_acc', best_test_acc)
        print(f"[Ori-HAR] Resume training from epoch {start_epoch}.")
    else:
        print("[Ori-HAR] Start training from scratch.")
        _remove_stale_checkpoint(last_parameter_path, 'last checkpoint')
        _remove_stale_checkpoint(best_parameter_path, 'best checkpoint')

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        random.seed(epoch)
        num_iter = int(max_train_batches) if max_train_batches is not None else len(train_loader)

        for step, data in enumerate(tqdm(train_loader)):
            if step >= num_iter:
                break

            vk_data, depth_data, mmwave_data, lidar_data, labels, modality_list = _move_har_batch_to_device(data, device)
            optimizer.zero_grad()
            outputs = model(vk_data, depth_data, mmwave_data, lidar_data, modality_list).float()
            loss = criterion(outputs, labels)
            if torch.isnan(loss):
                raise RuntimeError('[Ori-HAR] Encountered NaN loss during training.')

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            predict_y = torch.argmax(outputs, dim=1)
            epoch_loss += loss.item() * batch_size
            epoch_correct += (predict_y == labels).sum().item()
            epoch_samples += batch_size

        epoch_loss = epoch_loss / max(epoch_samples, 1)
        epoch_accuracy = epoch_correct / max(epoch_samples, 1)
        print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch + 1, epoch_accuracy, epoch_loss))

        if (epoch + 1) % 10 == 0:
            test_acc = har_test(
                model=model,
                tensor_loader=test_loader,
                criterion=criterion,
                device=device,
                val_random_seed=val_random_seed,
            )
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                torch.save(
                    _build_har_checkpoint(epoch, model, optimizer, scheduler, best_test_acc),
                    best_parameter_path,
                )
                print(f"[Ori-HAR] Updated best checkpoint with accuracy={test_acc:.8f}")

        scheduler.step()
        torch.save(
            _build_har_checkpoint(epoch, model, optimizer, scheduler, best_test_acc),
            last_parameter_path,
        )
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            epoch_parameter_path = os.path.join(save_dir, f'epoch_{epoch + 1:03d}.pth')
            torch.save(
                _build_har_checkpoint(epoch, model, optimizer, scheduler, best_test_acc),
                epoch_parameter_path,
            )

    final_checkpoint = _build_har_checkpoint(num_epochs - 1, model, optimizer, scheduler, best_test_acc)
    torch.save(final_checkpoint, archive_parameter_path)
    if not os.path.isfile(best_parameter_path):
        torch.save(final_checkpoint, best_parameter_path)
        print("[Ori-HAR] Best checkpoint was absent; saved final checkpoint as best.pth")
    return


def multi_test(model, tensor_loader, criterion, device, val_random_seed, max_batches=None):
    model.eval()
    combo_specs = _har_combo_specs()
    totals = {
        name: {'loss': 0.0, 'correct': 0}
        for name, _ in combo_specs
    }

    random.seed(val_random_seed)
    with torch.no_grad():
        seen_samples = 0
        for batch_idx, data in enumerate(tqdm(tensor_loader)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            vk_data, depth_data, mmwave_data, lidar_data, labels, _ = _move_har_batch_to_device(data, device)
            batch_size = labels.size(0)
            seen_samples += batch_size

            for name, modality_list in combo_specs:
                outputs = model(vk_data, depth_data, mmwave_data, lidar_data, modality_list).float()
                totals[name]['loss'] += criterion(outputs, labels).item() * batch_size
                predict_y = torch.argmax(outputs, dim=1)
                totals[name]['correct'] += (predict_y == labels).sum().item()

    dataset_size = seen_samples if max_batches is not None else len(tensor_loader.dataset)
    for name, _ in combo_specs:
        test_loss = totals[name]['loss'] / dataset_size
        test_accuracy = totals[name]['correct'] / dataset_size
        print(
            "modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format(
                name,
                test_loss,
                test_accuracy,
            )
        )
