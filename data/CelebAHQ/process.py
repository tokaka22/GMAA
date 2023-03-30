'''
step.1 map CelebA-HQ to CelebA
step.2 filter by valid AUs (the confidence from OpenFace https://github.com/TadasBaltrusaitis/OpenFace need be greater or equal to 0.95)
'''

import numpy as np
import copy
import cv2
import glob
import os 
import pandas as pd
from tqdm import tqdm
import shutil
import re
import argparse
from matplotlib import pyplot as plt


src_hq_path = '/beegfs/home/huyx/ADV/data/test-data/CelebAMask-HQ/CelebA-HQ-img'
dest_hq_path = 'data/CelebAHQ/CelebA-HQ-filter-img/images_aligned'
map_path = 'data/CelebAHQ/CelebA-HQ-to-CelebA-mapping.txt'
au_path = 'data/CelebAHQ/list_attr_HQ-filter-img.txt'

map_dict = {}
with open(map_path, "r") as f:
    lines = f.readlines()
    lines = lines[1:]
    for line in lines:
        line = line.strip()
        split = line.split()
        map_dict[split[0]] = 'frame_det_00_' + split[1].zfill(6)

valid_name_list = []
with open(au_path, "r") as f:
    lines = f.readlines()
    lines = lines[1:]
    for line in lines:
        line = line.strip()
        split = line.split()
        valid_name_list.append(split[0])

ct = 0
for src_name in os.listdir(src_hq_path):
    dest_name = map_dict[src_name]
    if dest_name in valid_name_list:
        src_path = os.path.join(src_hq_path, src_name)
        dest_path = os.path.join(dest_hq_path, dest_name)

        shutil.copy(src_path, dest_path)
        ct += 1

assert ct == 27255