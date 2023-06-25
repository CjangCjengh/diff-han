## Dependencies
---
```shell
cd DCN
./make.sh
cd ..
pip install -r requirements.txt
```
## Preparing dataset
---
```shell
python font2img.py --font_path font_folder --save_path save_folder --img_size 128 --chara_size 128
```
## Training
---
cfg/train_cfg.json:
```json
"data_dir": [
    "./save_folder/font_a",
    "./save_folder/font_b",
    "./save_folder/font_c"
],
"num_tokens": 445,
"num_features": 64,
"diffusion_steps": 1000,
"noise_schedule": "linear",
"image_size": 128,
"num_channels": 128,
"num_res_blocks": 3,
"lr": 0.0001,
"batch_size": 16,
"log_interval": 250,
"save_interval": 1000,
"train_step": 420000,
"attention_resolutions": "40, 20, 10",
"model_save_dir": "./trained_models",
"ids_path": "./han_ids.txt",
"glyph_path": "./glyphs.json"
```
Single gpu
```shell
python train.py --cfg_path cfg/train_cfg.json
```
Distributed training
```shell
mpiexec -n $NUM_GPUS python train.py --cfg_path cfg/train_cfg.json
```
## Inference
---
cfg/test_cfg.json:
```json
"dropout": 0.1,
"num_tokens": 445,
"num_features": 64,
"num_fonts": 3, // number of data_dir
"diffusion_steps": 1000,
"noise_schedule": "linear",
"image_size": 128,
"num_channels": 128,
"num_res_blocks": 3,
"batch_size": 5,
"attention_resolutions": "40, 20, 10",
"use_ddim": true,
"timestep_respacing": "ddim25",
"model_path": "./trained_models/model.pt",
"gen_txt_file": "./gen_ids.txt",
"nth_font": 0, // nth font in data_dir
"img_save_path": "./result",
"ids_path": "./han_ids.txt",
"glyph_path": "./glyphs.json"
```
Then run
```shell
python inference.py --cfg_path cfg/test_cfg.json
```
