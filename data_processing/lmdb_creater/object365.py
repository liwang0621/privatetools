'''
Author       : Li Wang
Date         : 2022-04-21 16:58:45
LastEditors  : Li Wang
LastEditTime : 2022-05-12 19:06:20
FilePath     : /privatetools/data_processing/lmdb_creater/object365.py
Description  : 
'''
import numpy as np
import lmdb
import math
import os
import sys
import cv2
import json
import argparse
from tqdm import tqdm
from glob import glob
from loguru import logger
from IPython import embed

from multiprocessing import Process, Pool, Manager

sys.path.append('../')
from annos_processing.utils import load_annos

# def xywh2xyxy(box):
#     out = []
#     out.append(box[0])
#     out.append(box[1])
#     out.append(box[0] + box[2])
#     out.append(box[1] + box[3])
#     return out

# def create(img_names, img_dir, save_root):
#     save_lmdb_dir = os.path.join(save_root, "lmdb")
#     if not os.path.isdir(save_lmdb_dir):
#         os.makedirs(save_lmdb_dir)

#     env = lmdb.open(save_lmdb_dir, map_size=int(1e12))
#     txn = env.begin(write=True)
#     for img_idx in tqdm(range(0, len(img_names))):
#         # img_name, img_anno = img_names[img_idx]
#         img_name = img_names[img_idx]['img_name']
#         img_path = os.path.join(img_dir, img_name)
#         if ' ' in img_name:
#             logger.info('there is a space in the path')
#             img_name = img_name.replace(' ', '_')
#         img = cv2.imread(img_path)
#         if img is None:
#             logger.info('there is a problom!!!!!!!')
#             continue
#         img_data = cv2.imencode(".jpg", img)[1].tobytes()
#         txn.put(img_name.encode(), img_data)

#     txn.commit()
#     env.close()

# if __name__ == "__main__":
#     img_root = '/home/liwang/data/object365/train'
#     save_root = '/home/liwang/data/general_det/object365/train'

#     anno_path = '/home/liwang/data/general_det/object365/train/anno.txt'
#     annos = load_annos(anno_path)

#     create(annos, img_root, save_root)



def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def image_worker(args):
    data, label, queue = args
    if os.path.exists(data):
        # with open(data, 'rb') as f:
        #     imageBin = f.read()
        img = cv2.imread(img_path)
        imageBin = cv2.imencode(".jpg", img)[1].tobytes()
        # with open(labelpath, "r", encoding="utf-8") as l:
        #     label = l.read()
        # from pdb import set_trace; set_trace()
        label = label.encode()
        queue_data = dict(imageBin=imageBin, label=label)
        queue.put(queue_data)
    else:
        logger.info(f'{data} does not exist')


def image_reader(labellist, datalist, queue, pool):
    pool.map(image_worker, iter([(datalist[i], labellist[i], queue) for i in range(len(datalist))]))


def lmdb_writer(outputPath, max_size, queue, datalen):
    env = lmdb.open(outputPath, map_size=max_size)
    cache = {}
    cnt = 1
    while True:
        data = queue.get()
        if data != 'Done':
            imageBin = data["imageBin"]
            label = data["label"]

            cache[label] = imageBin

            if cnt % 5000 == 0:
                writeCache(env, cache)
                cache = {}
                logger.info(f'Written {cnt}/ {datalen}')
            cnt += 1
        else:
            nSamples = cnt - 1
            cache['num-samples'.encode()] = str(nSamples).encode()
            writeCache(env, cache)
            logger.info(f'Created {outputPath.split("/")[-1]} dataset with {nSamples} samples')
            break


def main(labellist, datalist, outputPath, max_size):
    manager = Manager()
    queue = manager.Queue(maxsize=50000000)
    writer_process = Process(target=lmdb_writer, name='lmdb writer',
                             args=(outputPath, max_size, queue, len(datalist)), daemon=True)
    writer_process.start()
    logger.info(f'lmdb writer is started with PID: {writer_process.pid}')
    pool = Pool(processes=8, maxtasksperchild=1000000)
    try:
        image_reader(labellist, datalist, queue, pool)
        queue.put('Done')
        pool.close()
        pool.join()
        writer_process.join()
    except Exception as e:
        logger.exception(f"Error: {e}")
        pool.terminate()
        pool.join()

if __name__ == "__main__":
    anno_path = '/home/liwang/data/general_det/object365/train/anno.txt'
    annos = load_annos(anno_path)

    img_root = '/home/liwang/data/object365/train'
    outputPath = '/home/liwang/data/general_det/object365/train/lmdb'
    # os.makedirs(outputPath, exist_ok=True)

    labellist = []
    datalist = []
    for img_idx in tqdm(range(0, len(annos))):
        # img_name, img_anno = img_names[img_idx]
        img_name = annos[img_idx]['img_name']
        img_path = os.path.join(img_root, img_name)
        labellist.append(img_name)
        datalist.append(img_path)
    del annos
    # from pdb import set_trace; set_trace()
    main(labellist, datalist, outputPath, max_size=int(1e12))