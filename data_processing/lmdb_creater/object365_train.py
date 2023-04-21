'''
Author       : Li Wang
Date         : 2022-04-21 16:58:45
LastEditors  : wang.li
LastEditTime : 2023-01-10 10:01:07
FilePath     : /privatetools/data_processing/lmdb_creater/object365_train.py
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
from ipdb import set_trace


sys.path.append('../')
from annos_processing.utils import load_annos, get_filelist


def create(img_names, img_dir, save_root):
    save_lmdb_dir = os.path.join(save_root, "lmdb")
    if not os.path.isdir(save_lmdb_dir):
        os.makedirs(save_lmdb_dir)

    env = lmdb.open(save_lmdb_dir, map_size=int(1e12))
    txn = env.begin(write=True)

    for img_idx in tqdm(range(0, len(img_names))):
        # img_name, img_anno = img_names[img_idx]
        img_name = img_names[img_idx]['img_name']

        img_path = os.path.join(img_dir, img_name)
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
    # img_root = '/home/liwang/data/deepfashion2/validation/image'
    # save_root = '/home/liwang/data/general_det/deepfashion2/validation'
    # create(annos, img_root, save_root)


    # lmdb_path = '/home/liwang/data/general_det/object365/train/lmdb'
    # env_db = lmdb.open(lmdb_path, readonly=True, lock=False) 
    # txn = env_db.begin()

    # for anno in tqdm(annos):
    #     img_name = anno['img_name']
    #     data = txn.get(img_name.encode())
    #     data = bytearray(data)
    #     img = cv2.imdecode(np.asarray(data, dtype=np.uint8),
    #                     cv2.IMREAD_COLOR)
    #     print(img.shape)




    root = '/home/liwang/data/product_data/检测训练数据/images/'

    Filelist = []
    get_filelist(root, Filelist, search_exts=['.jpeg'])
    logger.info(f'Files num: {len(Filelist)}')


    img_names = []
    for img_path in Filelist:
        img_name = img_path.split(root)[-1]
        img_names.append({'img_name': img_name})

    set_trace()
    create(img_names, root, '/home/liwang/data/product_det/dianshang_product')





