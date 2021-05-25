import os
import json
import argparse
import requests
import shutil
from PIL import Image
from datetime import datetime
import multiprocessing
import multiprocessing as mp


parser = argparse.ArgumentParser(description='根据txt标签文件，裁剪菜品图像')
parser.add_argument('--data_dir', type=str, default=r'W:\项目\菜品识别\菜品数据集\项目点数据\山东济南超算中心\data\2021-04-01_2021-05-11\test')
parser.add_argument('--coor_tran', type=bool, default=True)
args = parser.parse_args()

data_path = os.path.join(args.data_dir, 'imgs')
crop_path = os.path.join(args.data_dir, 'crop_imgs')

''' Remove a single image from the child folder to the parent folder.
    Params: 
        --data_dir: path of the parent directory
    Returns:
        Null
    Author:
        Natasha
    Date:
        2021/05/12
'''

def get_files_recursive(roots):
    imgs = []
    for parent, _, _ in os.walk(roots):
        for child, _, files in os.walk(parent):
            for file in files:
                if len(files) == 1: pass
                    # move to parent folder
                    if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                        # imgs.append(os.path.join(root, file))
                        print(f"Child: {child},\n File: {file}\n\n")
                        shutil.move(child+"\\"+file, parent+"\\"+file)
                        os.rmdir(child)
    
    # return imgs


if __name__ == "__main__":
    get_files_recursive(args.data_dir)

