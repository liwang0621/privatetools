'''
Author       : Li Wang
Date         : 2021-12-28 07:17:18
LastEditors  : wang.li
LastEditTime : 2023-01-06 16:00:26
FilePath     : /privatetools/data_processing/lmdb_vis/lmdb_vis.py
Description  : 
'''
import os
import cv2
import sys
import lmdb
import argparse
import numpy as np
from tqdm import tqdm

sys.path.append('../')
from pseudo_annotation.detector import Detector
from annos_processing.utils import load_annos

from IPython import embed
from pdb import set_trace


def render_result(img, bboxes, classes):
    color = (172, 47, 117)
    for bbox in bboxes:
        x1, y1, x2, y2, class_idx = bbox
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    for bbox in bboxes:
        x1, y1, x2, y2, class_idx = bbox
        try:
            class_name = classes[class_idx]
        except:
            # embed()
            set_trace()
        y_text = y1 + 20
        render_text = "{}:{:3.2f}".format(class_name, class_idx)
        cv2.putText(img, render_text, (int(x1), int(y_text)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

def get_detector(device='cuda'):
    # det config
    config_path = '/home/liwang/projects/mmdetection/configs/lw/server/gfl_swin-s_fpn_1x_widerface_FDDB_4kface_scaleaware_640_256_256_CLR.py'
    checkpoint_path = '/home/liwang/work_dirs/face_det/server/gfl_swin-s_fpn_1x_widerface_FDDB_4kface_scaleaware_640_256_256_CLR/epoch_24.pth'
    min_score = 0.5

    detector = Detector(config_path, checkpoint_path, min_score, device)
    return detector

def parse_args():
    parser = argparse.ArgumentParser(
        description='input lmdb anno')
    parser.add_argument(
        '--lmdb',
        help='lmdb', default='/home/liwang/data/face_det/dahuzi/lmdb')
    parser.add_argument(
        '--anno',
        help='anno.txt', default='/home/liwang/data/face_det/dahuzi/anno1.txt')
    parser.add_argument(
        '--save_dir',
        help='save_dir', default='/home/liwang/data/face_det/dahuzi/vis')
    parser.add_argument(
        '--det',
        action='store_true',
        help='shifou shiyong det')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    image_list = []
    with open(args.anno, 'r') as f:
        for line in f.readlines():
            image_list.append(line.strip())


    env_db = lmdb.open(args.lmdb, readonly=True, lock=False) 
    txn = env_db.begin()

    if args.det:
        detector = get_detector()

    annos =load_annos(args.anno)
    for anno in tqdm(annos):
        name = anno['img_name']
        value = txn.get(name.encode())
        data = bytearray(value)
        img = cv2.imdecode(np.asarray(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        bboxes = anno['bboxes']
        # classes = ['Face']
        classes = {
            # '人',
            # '动物',
            # '食物',
            # '服饰',
            # '交通工具',
            # '乐器',
            # '运动器械',
            # '电子产品',
            # '办公用品',
            # '家具家电',
            # '厨具',
            # '浴室用品',
            # '首饰',
            # '化妆品',
            # '玩具',
            # 1:'people',
            # 2:'animals',
            # 3:'food',
            # 4:'dress',
            # 5:'Vehicle ',
            # 6:'instruments',
            # 7:'Sports equipment ',
            # 8:'Electronic products ',
            # 9:'Office supplies ',
            # 10:'Home Appliances ',
            # 11:'kitchen utensils and appliances',
            # 12:'Bathroom supplies ',
            # 13:'jewelry',
            # 14:'Cosmetics ',
            # 15:'toys',
            # -1:'ignore'

            0: 'Ignor',
            1: 'Clothes',
            2: 'Toys',
            3: 'Makeup',
            4: 'Furniture',
            5: 'Jewelry',
            6: 'Others',
            7: 'Bags',
            8: 'Electrical',
            9: 'Shoes',
            10: 'Accessories',
            11: 'Food',
            12: 'Stationery',
            13: 'Empty'
        }

        render_result(img, bboxes, classes)
        # embed()
        name = name.replace('/', '_')
        save_path = os.path.join(args.save_dir, name)
        save_path = save_path + '.jpg'
        cv2.imwrite(save_path, img)



if __name__=='__main__':
    main()