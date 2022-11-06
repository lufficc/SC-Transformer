import math

import os
# import pickle
import pickle5 as pickle
from typing import List

import cv2
import torchvision.transforms.functional
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
from torchvision.io.image import ImageReadMode

from utils.distribute import synchronize, is_main_process


def image_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f).convert('RGB')
        return img


def prepare_gebd_annotations(cfg, root, name, split):
    frame_per_side = cfg.INPUT.FRAME_PER_SIDE
    ds = cfg.INPUT.DOWNSAMPLE
    dynamic_downsample = cfg.INPUT.DYNAMIC_DOWNSAMPLE
    min_change_dur = 0.3

    num_annotators = cfg.INPUT.ANNOTATORS if 'train' in split else 1

    if name == 'GEBD':
        ann_path = os.path.join('data', f'k400_mr345_{split}_min_change_duration0.3.pkl')
    elif name == 'TAPOS':
        ann_path = os.path.join('data', f'TAPOS_for_GEBD_{split}.pkl')
    else:
        raise NotImplemented

    filename = '{}_{}-cache-fps{}-ds{}.pkl'.format(name, split, frame_per_side, f'_dynamic{ds}' if dynamic_downsample else ds)
    if cfg.INPUT.END_TO_END:
        filename = 'end_to_end{}_'.format(cfg.INPUT.SEQUENCE_LENGTH) + filename

    if num_annotators > 1:
        filename = 'top{}_'.format(num_annotators) + filename

    cache_path = os.path.join('data', 'caches', filename)

    if is_main_process() and not os.path.exists(cache_path):
        with open(ann_path, 'rb') as f:
            dict_train_ann = pickle.load(f, encoding='lartin1')

        annotations = []
        neg = 0
        pos = 0

        for v_name in dict_train_ann.keys():
            v_dict = dict_train_ann[v_name]
            # {
            #     "num_frames": 150,
            #     "path_video": "air_drumming/tdpHu69TU5w_000004_000014.mp4",
            #     "fps": 15.0,
            #     "video_duration": 10.0,
            #     "path_frame": "air_drumming/tdpHu69TU5w_000004_000014",
            #     "f1_consis": [
            #         0.57264957264957,
            #         0.56666666666667,
            #         0.62230769230769,
            #         0.5777777777777799,
            #         0.53923076923077,
            #     ],
            #     "f1_consis_avg": 0.575726495726496,
            #     "substages_myframeidx": [
            #         [4.0, 14.0, 59.0, 89.0, 103.0, 141.0],
            #         [136.0],
            #         [12.0, 49.0, 86.0, 101.0, 133.0],
            #         [132.0, 143.0],
            #         [18.0, 33.0, 61.0, 93.0, 108.0, 135.0, 144.0],
            #     ],
            #     "substages_timestamps": [
            #         [0.301975, 0.95043, 3.98103, 5.98768, 6.87495, 9.44118],
            #         [9.11759],
            #         [0.8174, 3.30593, 5.75596, 6.77272, 8.8934],
            #         [8.85963, 9.55091],
            #         [1.20288, 2.20528, 4.10984, 6.21488, 7.21728, 9.0216, 9.62304],
            #     ],
            # }

            fps = v_dict['fps']
            f1_consis = v_dict['f1_consis']
            path_frame = v_dict['path_frame']
            video_duration = v_dict['video_duration']

            video_dir = os.path.join(root, path_frame)
            if not os.path.exists(video_dir):
                continue
            vlen = len(os.listdir(video_dir))

            if dynamic_downsample:
                downsample = max(math.ceil(fps / ds), 1)
            else:
                downsample = ds

            # select the annotation with highest f1 score
            # highest = np.argmax(f1_consis)
            # change_indices = v_dict['substages_myframeidx'][highest]
            # change_timestamps = v_dict['substages_timestamps'][highest]
            selected_annotators = np.argsort(f1_consis)[::-1][:num_annotators]

            for annotator_idx, annotator in enumerate(selected_annotators):
                if annotator_idx >= 1 and v_dict['f1_consis_avg'] > f1_consis[annotator]:
                    continue

                change_indices = v_dict['substages_myframeidx'][annotator]

                if cfg.INPUT.END_TO_END:
                    selected_indices = np.linspace(1, vlen, cfg.INPUT.SEQUENCE_LENGTH, dtype=int)
                    # selected_indices = np.arange(1, vlen + 1, ds, dtype=int)

                    half_dur_2_nframes = min_change_dur * fps / 2.

                    labels = []
                    for i in selected_indices:
                        labels.append(0)
                        for change in change_indices:
                            if change - half_dur_2_nframes <= i <= change + half_dur_2_nframes:
                                labels.pop()  # pop '0'
                                labels.append(1)
                                break

                    assert len(selected_indices) <= cfg.INPUT.SEQUENCE_LENGTH

                    if len(selected_indices) < cfg.INPUT.SEQUENCE_LENGTH:
                        offset_length = cfg.INPUT.SEQUENCE_LENGTH - len(selected_indices)
                        pad = -np.ones((offset_length,), dtype=int)
                        selected_indices = np.concatenate((selected_indices, pad))
                        labels += [0] * offset_length

                    assert len(labels) == len(selected_indices)
                    record = {
                        'folder': path_frame,
                        'block_idx': selected_indices.tolist(),
                        'label': labels,
                        'vid': v_name
                    }

                    annotations.append(record)
                else:
                    # (float)num of frames with min_change_dur/2
                    half_dur_2_nframes = min_change_dur * fps / 2.

                    start_offset = 1
                    selected_indices = np.arange(start_offset, vlen, downsample)

                    # should be tagged as positive(bdy), otherwise negative(bkg)
                    GT = []
                    for i in selected_indices:
                        GT.append(0)
                        for change in change_indices:
                            if change - half_dur_2_nframes <= i <= change + half_dur_2_nframes:
                                GT.pop()  # pop '0'
                                GT.append(1)
                                break

                    for idx, (current_idx, lbl) in enumerate(zip(selected_indices, GT)):
                        record = {}
                        shift = np.arange(-frame_per_side, frame_per_side)
                        shift[shift >= 0] += 1
                        shift = shift * downsample
                        block_idx = shift + current_idx
                        block_idx[block_idx < 1] = 1
                        block_idx[block_idx > vlen] = vlen
                        block_idx = block_idx.tolist()

                        record['folder'] = path_frame
                        record['current_idx'] = current_idx
                        record['block_idx'] = block_idx
                        record['label'] = lbl
                        record['vid'] = v_name
                        annotations.append(record)
                        if lbl == 0:
                            neg += 1
                        else:
                            pos += 1

        print(f'Split: {split}, GT: {len(dict_train_ann)}, Annotations: {len(annotations)}, Num pos: {pos}, num neg: {neg}, total: {pos + neg}')

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(annotations, f)

    synchronize()
    with open(cache_path, 'rb') as f:
        annotations = pickle.load(f)

    if is_main_process():
        print(f'Loaded from {cache_path}')

    return annotations


class ToFloat:
    def __call__(self, pic):
        return pic.to(torch.float32).div_(255)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class GEBDDataset(Dataset):
    def __init__(self, cfg, root, name="GEBD", split='train',
                 template='image_{:05d}.jpg',
                 train=True):
        annotations = prepare_gebd_annotations(cfg, root, name, split)

        self.ann_path = os.path.join('data', f'k400_mr345_{split}_min_change_duration0.3.pkl')
        self.cfg = cfg
        self.root = root
        self.name = name
        self.split = split
        self.template = template
        self.train = train
        self.annotations = annotations
        self.size = cfg.INPUT.RESOLUTION
        self.use_aug = cfg.INPUT.ARGUMENT

        transform = [transforms.Resize((self.size, self.size))]

        if train and self.use_aug:
            transform += [
                # transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(p=0.5),
            ]

        transform += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]

        self.transform = transforms.Compose(transform)

        if is_main_process():
            print('Split: {}'.format(split))
            print('Input Image Resolution: {}'.format(self.size))
            print('Use argument: {}, [{}]'.format(self.use_aug, self.transform))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        item = self.annotations[index]
        vid = item['vid']
        block_indices = item['block_idx']
        folder = item['folder']

        flip = torch.rand(1) < 0.5
        if self.train and self.use_aug:  # fix flip or not in one video
            self.transform.transforms[1].p = 1.0 if flip else 0.0

        imgs = torch.zeros(len(block_indices), 3, self.size, self.size, dtype=torch.float32)
        frame_masks = torch.ones(len(block_indices), dtype=torch.bool)
        use_frame_masks = False
        for i, frame_idx in enumerate(block_indices):
            if frame_idx == -1:
                frame_masks[i] = False
                continue

            img = image_loader(os.path.join(self.root, folder, self.template.format(frame_idx)))
            imgs[i] = self.transform(img)

        sample = {
            'imgs': imgs,
            'labels': torch.tensor(item['label'], dtype=torch.int64),
            'vid': vid,
        }
        if self.cfg.INPUT.END_TO_END:
            sample['frame_indices'] = torch.tensor(block_indices)
            if use_frame_masks:
                sample['frame_masks'] = frame_masks
        else:
            current_idx = item['current_idx']
            sample['frame_idx'] = current_idx
            sample['path'] = os.path.join(self.root, folder, self.template.format(current_idx))

        return sample
