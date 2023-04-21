'''
Author       : Li Wang
Date         : 2022-04-21 16:58:45
LastEditors  : wang.li
LastEditTime : 2022-08-08 11:31:01
FilePath     : /privatetools/data_processing/lmdb_creater/spu_comment.py
Description  : 
'''

import numpy as np
import lmdb
import math
import os
import cv2
import sys
import json
import argparse
from tqdm import tqdm
from glob import glob
from loguru import logger
from IPython import embed

sys.path.append('../')
from annos_processing.utils import load_annos

batch_size = 100


def create(img_names, datalist, outputPath):
    save_lmdb_dir = outputPath
    if not os.path.isdir(save_lmdb_dir):
        os.makedirs(save_lmdb_dir)

    env = lmdb.open(save_lmdb_dir, map_size=int(1e12))
    txn = env.begin(write=True)

    for img_idx in tqdm(range(0, len(img_names))):
        # img_name, img_anno = img_names[img_idx]
        img_name = img_names[img_idx]

        img_path = datalist[img_idx]
        if ' ' in img_name:
            logger.info('there is a space in the path')
            img_name = img_name.replace(' ', '_')
        img = cv2.imread(img_path)
        if img is None:
            logger.info('there is a problom!!!!!!!')
            continue
        img_data = cv2.imencode(".jpg", img)[1].tobytes()
        txn.put(img_name.encode(), img_data)

    txn.commit()
    env.close()


if __name__ == "__main__":
    labellist = []
    datalist = []
    # for img_path in glob('/home/liwang/data/shopee_shoes/spu_comment/images/*'):
    #     datalist.append(img_path)
    #     name = os.path.basename(img_path)
    #     labellist.append(name)

    img_root = '/home/liwang/data/shopee_shoes/spu_comment/shoes_with_leg/shoes_with_leg_1'
    # annos = load_annos('/home/liwang/data/foot_det/shopee_shoes/anno_v2.txt')
    for name in os.listdir(img_root):
        # name = anno['img_name']
        img_path = os.path.join(img_root, name)
        datalist.append(img_path)
        labellist.append(name)

    img_root = '/home/liwang/data/shopee_shoes/spu_comment/shoes_with_leg/shoes_with_leg_2'
    # annos = load_annos('/home/liwang/data/foot_det/shopee_shoes/anno_v1.txt')
    for name in os.listdir(img_root):
        # name = anno['img_name']
        img_path = os.path.join(img_root, name)
        datalist.append(img_path)
        labellist.append(name)

    outputPath = '/home/liwang/data/foot_det/shopee_shoes/shoes_with_leg_lmdb'
    create(labellist, datalist, outputPath)