import csv
import os
import scipy.io as scio
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def resolve_modality_dir(modality):
    return 'rgb' if modality == 'vk' else modality


def load_manifest_rows(manifest_path):
    if not manifest_path:
        return None
    rows = {}
    with open(manifest_path, 'r', newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            key = (row['scene'], row['subject'], row['action'], int(row['idx']))
            rows[key] = row
    print(f'[Ori-HPE Dataset] Loaded manifest | path={manifest_path} | samples={len(rows)}')
    return rows


VK_SKELETON_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


def normalize_vk_points(points, width=640, height=480):
    coords = np.array(points, dtype=np.float32)
    if coords.ndim == 1:
        if coords.size % 2 == 0:
            coords = coords.reshape(-1, 2)
        elif coords.size % 3 == 0:
            coords = coords.reshape(-1, 3)
        else:
            raise ValueError(f'Unexpected VK point shape: {coords.shape}')
    if coords.shape[-1] < 2:
        raise ValueError(f'Unexpected VK point shape: {coords.shape}')

    xy = coords[:, :2].copy()
    visibility = np.ones((xy.shape[0],), dtype=np.float32)
    if coords.shape[-1] >= 3:
        visibility = np.nan_to_num(coords[:, 2], nan=0.0, posinf=0.0, neginf=0.0)

    finite_mask = np.isfinite(xy).all(axis=1)
    visibility = visibility * finite_mask.astype(np.float32)
    xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)

    max_x = float(np.max(np.abs(xy[:, 0]))) if xy.size else 0.0
    max_y = float(np.max(np.abs(xy[:, 1]))) if xy.size else 0.0

    min_x = float(np.min(xy[:, 0])) if xy.size else 0.0
    min_y = float(np.min(xy[:, 1])) if xy.size else 0.0

    if max_x <= 2.0 and max_y <= 2.0:
        if min_x < 0.0 or min_y < 0.0:
            xy[:, 0] = (xy[:, 0] + 1.0) * 0.5 * (width - 1)
            xy[:, 1] = (xy[:, 1] + 1.0) * 0.5 * (height - 1)
        else:
            xy[:, 0] *= (width - 1)
            xy[:, 1] *= (height - 1)
    elif max_x <= 256.0 and max_y <= 256.0:
        xy[:, 0] *= float(width) / 224.0
        xy[:, 1] *= float(height) / 224.0

    xy[:, 0] = np.clip(xy[:, 0], 0, width - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, height - 1)
    return xy, visibility


def vk_points_to_pseudo_rgb(points, width=640, height=480):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    xy, visibility = normalize_vk_points(points, width=width, height=height)

    point_colors = [
        (255, 255, 255), (255, 220, 180), (255, 220, 180), (255, 200, 120), (255, 200, 120),
        (120, 220, 255), (120, 220, 255), (80, 180, 255), (80, 180, 255), (40, 140, 255), (40, 140, 255),
        (120, 255, 180), (120, 255, 180), (80, 220, 120), (80, 220, 120), (40, 180, 80), (40, 180, 80),
    ]

    for start, end in VK_SKELETON_BONES:
        if start >= len(xy) or end >= len(xy):
            continue
        if visibility[start] <= 0.01 or visibility[end] <= 0.01:
            continue
        p1 = tuple(np.round(xy[start]).astype(np.int32))
        p2 = tuple(np.round(xy[end]).astype(np.int32))
        cv2.line(canvas, p1, p2, color=(180, 255, 180), thickness=4, lineType=cv2.LINE_AA)

    for idx, point in enumerate(xy):
        if visibility[idx] <= 0.01:
            continue
        center = tuple(np.round(point).astype(np.int32))
        color = point_colors[idx % len(point_colors)]
        cv2.circle(canvas, center, radius=6, color=color, thickness=-1, lineType=cv2.LINE_AA)

    canvas = cv2.GaussianBlur(canvas, (5, 5), 0)
    return canvas.astype(np.float32)

def decode_config(config):
    all_subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14',
                    'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28',
                    'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']
    all_actions = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14',
                   'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27']
    train_form = {}
    val_form = {}
    # Limitation to actions (protocol)
    if config['protocol'] == 'protocol1':  # Daily actions
        actions = ['A02', 'A03', 'A04', 'A05', 'A13', 'A14', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A27']
    elif config['protocol'] == 'protocol2':  # Rehabilitation actions:
        actions = ['A01', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A15', 'A16', 'A24', 'A25', 'A26']
    else:
        actions = all_actions
    # Limitation to subjects and actions (split choices)
    if config['split_to_use'] == 'random_split':
        rs = config['random_split']['random_seed']
        ratio = config['random_split']['ratio']
        for action in actions:
            np.random.seed(rs)
            idx = np.random.permutation(len(all_subjects))
            idx_train = idx[:int(np.floor(ratio*len(all_subjects)))]
            idx_val = idx[int(np.floor(ratio*len(all_subjects))):]
            subjects_train = np.array(all_subjects)[idx_train].tolist()
            subjects_val = np.array(all_subjects)[idx_val].tolist()
            for subject in all_subjects:
                if subject in subjects_train:
                    if subject in train_form:
                        train_form[subject].append(action)
                    else:
                        train_form[subject] = [action]
                if subject in subjects_val:
                    if subject in val_form:
                        val_form[subject].append(action)
                    else:
                        val_form[subject] = [action]
            rs += 1
    elif config['split_to_use'] == 'cross_scene_split':
        subjects_train = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
                          'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20',
                          'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30']
        subjects_val = ['S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']
        for subject in subjects_train:
            train_form[subject] = actions
        for subject in subjects_val:
            val_form[subject] = actions
    elif config['split_to_use'] == 'cross_subject_split':
        subjects_train = config['cross_subject_split']['train_dataset']['subjects']
        subjects_val = config['cross_subject_split']['val_dataset']['subjects']
        for subject in subjects_train:
            train_form[subject] = actions
        for subject in subjects_val:
            val_form[subject] = actions
    else:
        subjects_train = config['manual_split']['train_dataset']['subjects']
        subjects_val = config['manual_split']['val_dataset']['subjects']
        actions_train = config['manual_split']['train_dataset']['actions']
        actions_val = config['manual_split']['val_dataset']['actions']
        for subject in subjects_train:
            train_form[subject] = actions_train
        for subject in subjects_val:
            val_form[subject] = actions_val

    dataset_config = {'train_dataset': {'modality': config['modality'],
                                        'split': 'training',
                                        'data_form': train_form
                                        },
                      'val_dataset': {'modality': config['modality'],
                                      'split': 'validation',
                                      'data_form': val_form}}
    return dataset_config

class MetaFi_Database:
    def __init__(self, data_root):
        self.data_root = data_root
        self.scenes = {}
        self.subjects = {}
        self.actions = {}
        self.modalities = {}
        self.load_database()

    def load_database(self):
        for scene in sorted(os.listdir(self.data_root)):
            self.scenes[scene] = {}
            for subject in sorted(os.listdir(os.path.join(self.data_root, scene))):
                self.scenes[scene][subject] = {}
                self.subjects[subject] = {}
                for action in sorted(os.listdir(os.path.join(self.data_root, scene, subject))):
                    self.scenes[scene][subject][action] = {}
                    self.subjects[subject][action] = {}
                    if action not in self.actions.keys():
                        self.actions[action] = {}
                    if scene not in self.actions[action].keys():
                        self.actions[action][scene] = {}
                    if subject not in self.actions[action][scene].keys():
                        self.actions[action][scene][subject] = {}
                    for modality in ['depth', 'vk', 'lidar', 'mmwave', 'wifi-csi']:
                        data_dir = resolve_modality_dir(modality)
                        data_path = os.path.join(self.data_root, scene, subject, action, data_dir)
                        self.scenes[scene][subject][action][modality] = data_path
                        self.subjects[subject][action][modality] = data_path
                        self.actions[action][scene][subject][modality] = data_path
                        if modality not in self.modalities.keys():
                            self.modalities[modality] = {}
                        if scene not in self.modalities[modality].keys():
                            self.modalities[modality][scene] = {}
                        if subject not in self.modalities[modality][scene].keys():
                            self.modalities[modality][scene][subject] = {}
                        if action not in self.modalities[modality][scene][subject].keys():
                            self.modalities[modality][scene][subject][action] = data_path


class Domain_Invariant_Dataset(Dataset):
    def __init__(self, data_base, modality, split, data_form, manifest_rows=None, visual_source='vk'):
        self.data_base = data_base
        # self.modality = modality.split('|')
        self.modality = modality
        print(self.modality)
        for m in self.modality:
            assert m in ['vk','depth', 'lidar', 'mmwave', 'wifi-csi']
        self.split = split
        self.data_source = data_form
        self.manifest_rows = manifest_rows
        self.visual_source = visual_source
        self.data_list = self.load_data()

    def get_scene(self, subject):
        if subject in ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']:
            return 'E01'
        elif subject in ['S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']:
            return 'E02'
        elif subject in ['S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30']:
            return 'E03'
        elif subject in ['S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']:
            return 'E04'
        else:
            raise ValueError('Subject does not exist in this dataset.')

    def load_data(self):
        data_info = tuple()
        for subject, actions in self.data_source.items():
            print(subject, actions)
            for action in actions:
                frame_list = sorted(os.listdir(os.path.join(self.data_base.data_root, self.get_scene(subject), subject, action, 'mmwave')))
                frame_num = len(frame_list)
                for idx in range(frame_num):
                    frame_idx = int(frame_list[idx].split('.')[0].split('frame')[1]) - 1
                    manifest_key = (self.get_scene(subject), subject, action, frame_idx)
                    manifest_row = self.manifest_rows.get(manifest_key) if self.manifest_rows is not None else None
                    if self.manifest_rows is not None and manifest_row is None:
                        continue
                    data_dict = {'modality': self.modality,
                                    'scene': self.get_scene(subject),
                                    'subject': subject,
                                    'action': action,
                                    'gt_path': os.path.join(self.data_base.data_root, self.get_scene(subject), subject,
                                                            action, 'ground_truth.npy'),
                                    'idx': frame_idx
                                    }
                    for mod in self.modality:
                        data_dir = resolve_modality_dir(mod)
                        if mod == 'vk' and self.visual_source == 'rgb' and manifest_row is not None:
                            data_dict[mod+'_path'] = manifest_row['rgb_path']
                        elif mod == 'mmwave':
                            data_dict[mod+'_path'] = os.path.join(self.data_base.data_root, self.get_scene(subject), subject, action, data_dir, sorted(os.listdir(os.path.join(self.data_base.data_root, self.get_scene(subject), subject, action, data_dir)))[idx])
                        else:
                            data_dict[mod+'_path'] = os.path.join(self.data_base.data_root, self.get_scene(subject), subject, action, data_dir, sorted(os.listdir(os.path.join(self.data_base.data_root, self.get_scene(subject), subject, action, data_dir)))[frame_idx])
                    data_info += (data_dict,)
        return data_info

    def read_frame(self, frame):
        _mod, _frame = os.path.split(frame)
        _, mod = os.path.split(_mod)
        image_ext = os.path.splitext(frame)[1].lower() in ['.png', '.jpg', '.jpeg', '.bmp']
        if mod == 'rgb' or image_ext:
            if frame.lower().endswith('.npy'):
                data = np.load(frame, allow_pickle=False)
                if isinstance(data, np.ndarray) and data.ndim == 3 and data.shape[-1] in [1, 3]:
                    if data.shape[-1] == 1:
                        data = np.repeat(data, 3, axis=2)
                    data = data.astype(np.float32)
                else:
                    data = vk_points_to_pseudo_rgb(data)
            else:
                data = cv2.imread(frame)
                if data is None:
                    raise FileNotFoundError(f'Failed to read RGB image: {frame}')
                data = data.astype(np.float32)
        elif mod in ['infra1', 'infra2']:
            data = cv2.imread(frame)
        elif mod == 'depth':
            data = cv2.imread(frame)
        elif mod == 'lidar':
            with open(frame, 'rb') as f:
                raw_data = f.read()
                data = np.frombuffer(raw_data, dtype=np.float64)
                data = data.reshape(-1, 3)
        elif mod == 'mmwave':
            with open(frame, 'rb') as f:
                raw_data = f.read()
                data = np.frombuffer(raw_data, dtype=np.float64)
                data = data.copy().reshape(-1, 5)
                # data = data[:, :3]
        elif mod == 'wifi-csi':
            data = scio.loadmat(frame)['CSIamp']
            data[np.isinf(scio.loadmat(frame)['CSIamp'])] = np.nan
            for i in range(10):
                temp_col = data[:, :, i]
                nan_num = np.count_nonzero(temp_col != temp_col)
                if nan_num != 0:
                    temp_not_nan_col = temp_col[temp_col == temp_col]
                    temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()

            data = torch.tensor((data - np.min(data)) / (np.max(data) - np.min(data)))
            data = np.array(data)
        else:
            raise ValueError('Found unseen modality in this dataset.')
        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        gt_numpy = np.load(item['gt_path'])
        gt_torch = torch.from_numpy(gt_numpy)
        sample = {'modality': item['modality'],
                    'scene': item['scene'],
                    'subject': item['subject'],
                    'action': item['action'],
                    'idx': item['idx'],
                    'output': gt_torch[item['idx']]
                    }
        for mod in item['modality']:
            data_path = item[mod + '_path']
            if os.path.isfile(data_path):
                data_mod = self.read_frame(data_path)
                sample['input_'+mod] = data_mod
            else:
                raise ValueError('{} is not a file!'.format(data_path))
        return sample
        # sample = {'modality': ['vk', 'depth', 'lidar', 'mmwave'],
        #           'scene': 'E01',
        #           'subject': 'S02',
        #           'action': 'A01',
        #           'idx': 6,
        #           'output': torch_tensor(17x3),
        #           'input_vk': numpy_array(17x2),
        #           'input_depth': cv2_img(480x640x3),
        #           'input_lidar': numpy_array(480x640x3),
        #           'input_mmwave': numpy_array(480x640x3)
        #           }


def make_dataset(dataset_root, config):
    database = MetaFi_Database(dataset_root)
    config_dataset = decode_config(config)
    manifest_rows = load_manifest_rows(config.get('manifest_path'))
    visual_source = config.get('visual_source', 'vk')
    train_dataset = Domain_Invariant_Dataset(database, manifest_rows=manifest_rows, visual_source=visual_source, **config_dataset['train_dataset'])
    print('/n')
    val_dataset = Domain_Invariant_Dataset(database, manifest_rows=manifest_rows, visual_source=visual_source, **config_dataset['val_dataset'])
    return train_dataset, val_dataset

def make_dataloader(dataset, is_training, generator, batch_size, collate_fn):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=is_training,
        drop_last=is_training,
        generator=generator
    )
    return loader
