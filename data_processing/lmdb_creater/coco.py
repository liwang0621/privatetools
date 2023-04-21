'''
Author       : Li Wang
Date         : 2022-04-21 16:58:45
LastEditors  : wang.li
LastEditTime : 2022-05-29 15:03:17
FilePath     : /privatetools/data_processing/lmdb_creater/coco.py
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

def create(img_names, img_list, save_root):
    save_lmdb_dir = save_root
    if not os.path.isdir(save_lmdb_dir):
        os.makedirs(save_lmdb_dir)

    env = lmdb.open(save_lmdb_dir, map_size=int(1e12))
    txn = env.begin(write=True)

    for img_name, img_path in tqdm(zip(img_names, img_list)):
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
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--anno_dir", default='/home/liwang/data/4K-Face')
    # parser.add_argument("--img_dir", default='/home/liwang/data/4K-Face/images')
    # parser.add_argument("--save_root", default='/home/liwang/data/face_det/4K-Face')

    # args = parser.parse_args()

    # anno_path = '/home/liwang/data/general_det/deepfashion2/validation/anno_15class.txt'
    # annos = load_annos(anno_path)


    labellist = []
    datalist = []
    for img_path in glob('/home/liwang/data/coco/train2017/*'):
        datalist.append(img_path)
        name = os.path.basename(img_path)
        labellist.append(name)
    outputPath = '/home/liwang/data/foot_det/coco/train2017/lmdb'
    # main(labellist, datalist, outputPath, max_size=int(1e12))

    create(labellist, datalist, outputPath)

