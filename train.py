import os
import argparse
from utils import dist_util, logger
from utils.image_datasets import load_data
from utils.resample import create_named_schedule_sampler
from utils.script_util import (
    model_and_diffusion_defaults,
    args_to_dict,
    create_model_and_diffusion,
)
from utils.train_util import TrainLoop
import torch
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='./cfg/train_cfg.json',
                        help='config file path')
    parser = parser.parse_args()
    with open(parser.cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    cfg = create_cfg(cfg)
    train_step = cfg['train_step']
    ids_path = cfg['ids_path']
    glyph_path = cfg['glyph_path']
    cfg['num_fonts'] = len(cfg['data_dir'])

    dist_util.setup_dist()

    model_save_dir = cfg['model_save_dir']

    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    logger.configure(dir=model_save_dir, format_strs=['stdout', 'log', 'csv']) 

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(cfg, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(cfg['schedule_sampler'], diffusion)


    logger.log("creating data loader...")
    data = load_data(
        data_dir=cfg['data_dir'],
        batch_size=cfg['batch_size'],
        image_size=cfg['image_size'],
        ids_path=ids_path,
        glyph_path=glyph_path,
    )
    
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=cfg['batch_size'],
        microbatch=cfg['microbatch'],
        lr=cfg['lr'],
        ema_rate=cfg['ema_rate'],
        log_interval=cfg['log_interval'],
        save_interval=cfg['save_interval'],
        train_step=train_step,
        use_fp16=cfg['use_fp16'],
        fp16_scale_growth=cfg['fp16_scale_growth'],
        schedule_sampler=schedule_sampler,
        weight_decay=cfg['weight_decay']
    ).run_loop()


def create_cfg(cfg):
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=250,
        save_interval=20000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        stroke_path=None,
        attention_resolutions='40, 20, 10',
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cfg)
    return defaults


if __name__ == "__main__":
    main()
