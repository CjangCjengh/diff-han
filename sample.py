import argparse
import os

import numpy as np
import torch
import torch.distributed as dist

from utils import dist_util, logger
from utils.script_util import (
    model_and_diffusion_defaults,
    args_to_dict,
    create_model_and_diffusion,
)
from PIL import Image
import json


def img_pre_pros(img_path, image_size):
    pil_image = Image.open(img_path).resize((image_size, image_size))
    pil_image.load()
    pil_image = pil_image.convert("RGB")
    arr = np.array(pil_image)
    arr = arr.astype(np.float32) / 127.5 - 1
    return np.transpose(arr, [2, 0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='./cfg/test_cfg.json',
                        help='config file path')
    parser = parser.parse_args()
    with open(parser.cfg_path, 'r') as f:
        cfg = json.load(f)
    cfg = create_cfg(cfg)
    model_path = cfg['model_path']
    gen_txt_file = cfg['gen_txt_file']
    img_save_path = cfg['img_save_path']
    batch_size = cfg['batch_size']

    dist_util.setup_dist()

    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(cfg, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if cfg['use_fp16']:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    noise = None

    # gen txt
    glyph_dict = {}
    with open(cfg['glyph_path'], 'r', encoding='utf-8') as f:
        glyphs = json.load(f)
    for idx, glyph in enumerate(glyphs):
        glyph_dict[glyph] = idx + 3

    ids_dict = {}
    with open(cfg['ids_path'], 'r', encoding='utf-8') as f:
        for line in f.readlines():
            char, ids = line.strip().split('\t')
            ids_dict[char] = ids

    def divide(ids):
        new_ids=''
        for c in ids:
            if c in ids_dict:
                new_ids+=ids_dict[c]
            else:
                new_ids+=c
        return new_ids
    
    ids_seqs = []
    with open(gen_txt_file, 'r') as f:
        for ids in f.readlines():
            ids = ids.strip()
            new_ids=divide(ids)
            while new_ids!=ids:
                ids=new_ids
                new_ids=divide(ids)
            ids = [glyph_dict[c] if c in glyph_dict else glyph_dict['？'] for c in ids]
            ids = [0] + ids + [2]
            ids_seqs.append(str(ids)[1:-1])

    ch_idx = 0
    while ch_idx < len(ids_seqs):
        model_kwargs = {}
        model_kwargs["ids"] = ids_seqs[ch_idx:ch_idx+batch_size]

        sample_fn = (
            diffusion.p_sample_loop if not cfg['use_ddim'] else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (len(model_kwargs["ids"]), 3, cfg['image_size'], cfg['image_size']),
            clip_denoised=cfg['clip_denoised'],
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            noise=noise,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        for idx, img_sample in enumerate(sample):
            img = Image.fromarray(img_sample.cpu().numpy()).convert("RGB")
            img_name = "%05d.png" % (idx + ch_idx + 1)
            img.save(os.path.join(img_save_path, img_name))

        logger.log(f"created {ch_idx + len(sample)} samples")
        ch_idx += batch_size

    dist.barrier()
    logger.log("sampling complete")


def create_cfg(cfg):
    defaults = dict(
        clip_denoised=True,
        num_samples=100,
        batch_size=16,
        use_ddim=False,
        model_path="",
        cont_scale=1.0,
        sk_scale=1.0,
        sty_img_path="",
        stroke_path=None,
        attention_resolutions='40, 20, 10',
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cfg)
    return defaults


if __name__ == "__main__":
    main()
