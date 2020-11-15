import json
import os
import random
import shutil  
from tqdm import tqdm

ORI_DIR = "tmp/meishi_card/original"
SEG_DIR = "tmp/meishi_card/segmentation"

DEST_DIR = ["tmp/trainvaltest_meishi_card/train", "tmp/trainvaltest_meishi_card/val", "tmp/trainvaltest_meishi_card/test"]
DEST_SPLIT = [0.8, 0.9, 1]


def _create_dir(dirName):
    # Create target Directory if don't exist
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")


def create_dirs(saved_path=DEST_DIR):
    for path in saved_path:
        ori_dir = os.path.join(path, 'original')
        seg_dir = os.path.join(path, 'segmentation')

        _create_dir(ori_dir)
        _create_dir(seg_dir)
    print('Done creating directories')


create_dirs()

ori_paths = {}
for root, dirs, files in os.walk(ORI_DIR, topdown=False):
    for name in files:
        key = name.split('.')[0].replace(' ', '').lower()
        ori_paths[key] = os.path.join(root, name)

seg_paths = {}
for root, dirs, files in os.walk(SEG_DIR, topdown=False):
    for name in files:
        key = name.split('.')[0].replace(' ', '').lower()
        seg_paths[key] = os.path.join(root, name)

keys =  list(ori_paths.keys())
random.shuffle(keys)

i = 0
j = 0
total = len(keys)
for key in tqdm(keys):
    i = i + 1
    ori = ori_paths[key]
    seg = seg_paths[key]
    dest = DEST_DIR[j]
    if i > int(total*DEST_SPLIT[j]):
        j = j + 1
    shutil.copy(ori, dest + '/original/' + key + '.png')
    shutil.copy(seg, dest + '/segmentation/' + key + '.png')
