import os
import math
import random
import json

from PIL import Image
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from . import logger


def load_ids_dict(ids_path, glyph_path):
    logger.log("loading IDS data...")
    with open(glyph_path, 'r', encoding='utf-8') as f:
        glyphs = json.load(f)
    glyph_dict = {}
    for idx, glyph in enumerate(glyphs):
        glyph_dict[glyph] = idx + 3

    ids_dict = {}
    with open(ids_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            char, ids = line.strip().split('\t')
            ids = [glyph_dict[c] for c in ids]
            ids = [0] + ids + [2]
            ids_dict[char] = ids
    return ids_dict


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    deterministic=False,
    random_crop=False,
    random_flip=False,
    ids_dict=None,
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    
    all_files = _list_image_files_recursively(data_dir)
    all_files = [f for f in all_files if chr(int(os.path.basename(f).split('.')[0], 16)) in ids_dict]

    dataset = ImageDataset(
        image_size,
        all_files,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        ids_dict=ids_dict,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
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
        ids_dict=None,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.ids_dict = ids_dict

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

        out_dict = {}
        char = chr(int(os.path.basename(path).split('.')[0], 16))
        out_dict['ids'] = str(self.ids_dict[char])[1:-1]

        return np.transpose(arr, [2, 0, 1]), out_dict


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
