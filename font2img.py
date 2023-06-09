from PIL import Image,ImageDraw,ImageFont
import matplotlib.pyplot as plt
import os
import numpy as np
import pathlib
import argparse
from fontTools.ttLib import TTFont

def get_char_list_from_ttf(font_file):
    f_obj = TTFont(font_file)
    m_dict = f_obj.getBestCmap()
    
    unicode_list = []
    for key, _ in m_dict.items():
        unicode_list.append(key)

    char_list = [chr(ch_unicode) for ch_unicode in unicode_list]
    return char_list

def draw_single_char(ch, font, canvas_size):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = draw.textbbox((0, 0), ch, font=font)
    x_offset = (canvas_size - (x1 - x0)) // 2 - x0
    y_offset = (canvas_size - (y1 - y0)) // 2 - y0
    draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Obtaining characters from .ttf')
    parser.add_argument('--ttf_path', type=str, default='../ttf_folder',help='ttf directory')
    parser.add_argument('--save_path', type=str, default='../save_folder',help='images directory')
    parser.add_argument('--img_size', type=int, help='The size of generated images')
    parser.add_argument('--chara_size', type=int, help='The size of generated characters')
    args = parser.parse_args()

    data_dir = args.ttf_path
    data_root = pathlib.Path(data_dir)

    all_image_paths = list(data_root.glob('*.[to]tf'))
    all_image_paths = [str(path) for path in all_image_paths]
    total_num = len(all_image_paths)

    seq = list()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    for idx, item in enumerate(all_image_paths):
        print("{} / {} ".format(idx, total_num), item)
        src_font = ImageFont.truetype(item, size=args.chara_size)
        font_name = os.path.basename(item).split('.')[0]
        chars = get_char_list_from_ttf(item)
        
        for char in chars:
            try:
                img = draw_single_char(char, src_font, args.img_size)
            except:
                continue
            path_full = os.path.join(args.save_path, font_name)
            if not os.path.exists(path_full):
                os.mkdir(path_full)
            img.save(os.path.join(path_full, "%s.png" % hex(ord(char))))
