import os
import re
import math
import random
import json

from PIL import Image
from mpi4py import MPI
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from . import logger
from .ids_encoder import IDSEncoder


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    deterministic=False,
    random_crop=False,
    random_flip=False,
    ids_path=None,
    glyph_path=None,
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    
    ids_encoder = IDSEncoder(ids_path, glyph_path, 32)
    
    all_files = _list_image_files_recursively(data_dir)
    all_files = [f for f in all_files if chr(int(os.path.basename(f).split('.')[0], 16)) in ids_encoder.ids_dict]

    dataset = ImageDataset(
        image_size,
        all_files,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        ids_encoder=ids_encoder,
    )

    def collate_fn(batch):
        arr_list = []
        y_list = []
        y_mask_list = []
        max_len = max([len(y) for arr, y, y_mask in batch])
        for arr, y, y_mask in batch:
            arr_list.append(arr)
            if max_len > len(y):
                temp = np.zeros((max_len - len(y), y.shape[1], y.shape[2]), dtype=int)
                y_list.append(np.concatenate((y, temp), axis=0))
                y_mask_list.append(np.concatenate((y_mask, temp), axis=0))
            else:
                y_list.append(y)
                y_mask_list.append(y_mask)
        arr = np.stack(arr_list)
        y = np.stack(y_list)
        y_mask = np.stack(y_mask_list)
        arr = torch.from_numpy(arr)
        out_dict = {}
        out_dict['y'] = torch.from_numpy(y)
        out_dict['y_mask'] = torch.from_numpy(y_mask)
        return arr, out_dict

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True, collate_fn=collate_fn
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True, collate_fn=collate_fn
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in os.listdir(data_dir):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif os.path.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        ids_encoder=None,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.ids_encoder = ids_encoder

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with open(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1
        y, y_mask = self.ids_encoder.encode_char(chr(int(os.path.basename(path).split('.')[0], 16)))

        return np.transpose(arr, [2, 0, 1]), y, y_mask


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
