'''
Author       : Li Wang
Date         : 2022-04-11 18:46:02
LastEditors  : wang.li
LastEditTime : 2023-01-06 15:15:24
FilePath     : /privatetools/data_processing/annos_processing/utils.py
Description  : 
'''
import os
import cv2
import lmdb
import json
import random
import numpy as np
from tqdm import tqdm
from loguru import logger

def get_filelist(dir, Filelist, search_exts= ['.jpg']):
    newDir = dir
    if os.path.isfile(dir) and os.path.splitext(dir)[-1] in search_exts:
        Filelist.append(dir)
        # # 若只是要返回文件文，使用这个
        # Filelist.append(os.path.basename(dir))
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # 如果需要忽略某些文件夹，使用以下代码
            #if s == "xxx":
                #continue
            newDir=os.path.join(dir,s)
            get_filelist(newDir, Filelist, search_exts)
    return Filelist

def create_bbox_by_keypoints(keypoints):
    x1 = np.min(keypoints[:, 0])
    y1 = np.min(keypoints[:, 1])
    x2 = np.max(keypoints[:, 0])
    y2 = np.max(keypoints[:, 1])
    return [x1, y1, x2, y2]

def annostr2dict(annostr):
    info = annostr.strip().split()
    img_name = info[0]
    img_w = int(info[1])
    img_h = int(info[2])
    obj_num = int(info[3])
    bboxes = []
    labels = []
    
    for obj_idx in range(0, obj_num):
        x1 = int(float(info[4 + obj_idx * 5]))
        y1 = int(float(info[5 + obj_idx * 5]))
        x2 = int(float(info[6 + obj_idx * 5]))
        y2 = int(float(info[7 + obj_idx * 5]))
        label = int(float(info[8 + obj_idx * 5]))
        if x2 <= x1 or y2 <= y1:
            continue
        bbox = [x1, y1, x2, y2, label]
        
        labels.append(label - 1)
        bboxes.append(bbox)
    out = {
        'img_name': img_name,
        'bboxes': bboxes,
        'img_w': img_w,
        'img_h': img_h
    }
    return out

def load_annos(anno_path):
    annos = []
    with open(anno_path, "r") as f:
        for line in tqdm(f.readlines()):
            try:
                anno = annostr2dict(line)
            except:
                raise TypeError('{}'.format(line))
            annos.append(anno)
    return annos

def save_annos(annos, save_anno_path):
    with open(save_anno_path, "w") as f:
        for anno in tqdm(annos):
            img_name = anno['img_name']
            bboxes = anno['bboxes']
            img_w = anno['img_w']
            img_h = anno['img_h']
            img_anno_str = ' '
            for box in bboxes:
                assert len(box) == 5
                label =  box[4]
                img_anno_str += '{} {} {} {} {} '.format(float(box[0]), float(box[1]), float(box[2]), float(box[3]), int(label))
            img_anno = img_name + " {} {} {}".format(img_w, img_h, len(bboxes)) + img_anno_str

            f.writelines(img_anno + '\n')
    logger.info('anno saved in {}'.format(save_anno_path))
