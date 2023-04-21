'''
Author       : Li Wang
Date         : 2021-12-28 07:17:18
LastEditors  : wang.li
LastEditTime : 2022-06-27 10:15:24
FilePath     : /privatetools/data_processing/lmdb_vis/lmdb2img.py
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

def parse_args():
    parser = argparse.ArgumentParser(
        description='input lmdb anno')
    parser.add_argument(
        '--lmdb',
        help='lmdb', default='/home/liwang/data/foot_det/shopee_shoes/train_v2/lmdb')
    parser.add_argument(
        '--anno',
        help='anno.txt', default='/home/liwang/data/foot_det/shopee_shoes/anno_v2.txt')
    parser.add_argument(
        '--save_dir',
        help='save_dir', default='/home/liwang/data/foot_det/shopee_shoes/train_v2/images_5w')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    env_db = lmdb.open(args.lmdb, readonly=True, lock=False) 
    txn = env_db.begin()

    annos =load_annos(args.anno)[:50000]
    for anno in tqdm(annos):
        name = anno['img_name']
        # value = txn.get(name.encode())
        # data = bytearray(value)
        # img = cv2.imdecode(np.asarray(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        # name = name.replace('/', '_')
        # save_path = os.path.join(args.save_dir, name)
        # cv2.imwrite(save_path, img)
        img_source_path = os.path.join('/home/liwang/data/shopee_shoes/spu_comment/images_v2', name)
        if os.path.exists(img_source_path):
            cmd = 'cp {} {}/'.format(img_source_path, args.save_dir)
            # print(cmd)
            os.system(cmd)


if __name__=='__main__':
    main()