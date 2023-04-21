'''
Author       : Li Wang
Date         : 2021-12-27 02:33:55
LastEditors  : Li Wang
LastEditTime : 2022-03-10 03:38:29
FilePath     : /data_processing/pseudo_annotation/pesudo_face_det_multi.py
Description  : 
'''
import os
from re import I
import cv2
import lmdb
import torch
import numpy as np
from tqdm import tqdm
import multiprocessing
from multiprocessing import Manager
from loguru import logger

from IPython import embed
from detector import Detector, render_result
from mmcv.cnn import VGG

def get_detector(device='cuda'):
    # det config
    config_path = '/home/liwang/projects/mmdetection/configs/lw/server/gfl_swin-s_fpn_1x_widerface_FDDB_4kface_scaleaware_640_256_256_CLR.py'
    checkpoint_path = '/home/liwang/work_dirs/face_det/server/gfl_swin-s_fpn_1x_widerface_FDDB_4kface_scaleaware_640_256_256_CLR/epoch_24.pth'
    min_score = 0.3

    logger.info('*' * 10 + ' det config ' + '*' * 10)
    logger.info('config_path: {}'.format(config_path))
    logger.info('checkpoint_path: {}'.format(checkpoint_path))
    logger.info('min_score: {}'.format(min_score))
    logger.info('*'*31)

    detector = Detector(config_path, checkpoint_path, min_score, device)
    return detector

def worker(procnum, image_paths_worker, device, lmdb_path, return_dict):
    detector = get_detector(device)

    env_db = lmdb.open(lmdb_path, readonly=True, lock=False) 
    txn = env_db.begin()

    out = []
    for it in tqdm(image_paths_worker):
        name = it.split(' ')[0]
        value = txn.get(name.encode())
        data = bytearray(value)
        img = cv2.imdecode(np.asarray(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        result = detector.predict(img)
        out.append((it, result))
    return_dict[procnum] = out

    env_db.close()
    


def main():
    lmdb_path = '/home/liwang/data/face_fp/catdog/10.178.160.13:8000/catdog/lmdb'
    anno_save_path = '/home/liwang/data/face_fp/catdog/10.178.160.13:8000/catdog/anno_det.txt'
    image_list_path = '/home/liwang/data/face_fp/catdog/10.178.160.13:8000/catdog/anno.txt'

    logger.info('lmdb path: {}'.format(lmdb_path))
    logger.info('img list: {}'.format(image_list_path))
    logger.info('save path: {}'.format(anno_save_path))
    
    image_list = []
    with open(image_list_path, 'r') as f:
        for line in f.readlines():
            image_list.append(line.split(' ')[0])


    # image_list = image_list[:101]


    multithread = 8

    nr_lines = len(image_list)
    nr_liner_per_worker = np.ceil(nr_lines / multithread)


    manager = Manager()
    # return_list = manager.list() 也可以使用列表list
    return_dict = manager.dict()
    jobs = []
    for i in range(multithread):
        device = 'cuda:{}'.format(i%8)

        start = int(i * nr_liner_per_worker)                                               
        end = int(min(start + nr_liner_per_worker, nr_lines))
        image_paths_worker = image_list[start:end]

        p = multiprocessing.Process(target=worker, args=(i, image_paths_worker, device, lmdb_path, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
        
    result_dict = dict(return_dict)

    ff = open(anno_save_path.replace('.txt', '_raw_det.txt'), 'w')

    with open(anno_save_path, 'w') as f:
        for procnum in result_dict:
            for name, result in result_dict[procnum]:
                # anno str
                img_anno_str = ' '
                img_anno_str_raw = ' '
                for box in result:
                    assert len(box) == 6
                    img_anno_str += '{} {} {} {} {} '.format(box[2], box[3], box[4], box[5], 1)

                    img_anno_str_raw += '{} {} {} {} {} '.format(box[2], box[3], box[4], box[5], box[1])


                img_anno = name + " {}".format(len(result)) + img_anno_str
                img_anno_raw = name + " {}".format(len(result)) + img_anno_str_raw
                f.writelines(img_anno + '\n')
                ff.writelines(img_anno_raw + '\n')

    ff.close()

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
    # demo()
