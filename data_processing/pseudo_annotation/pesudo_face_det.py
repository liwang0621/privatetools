'''
Author       : Li Wang
Date         : 2021-12-22 03:58:38
LastEditors  : Li Wang
LastEditTime : 2021-12-22 04:44:58
FilePath     : /data_processing/pseudo-annotation/pesudo_face_det.py
Description  : 
'''
import os
import cv2
import lmdb
import numpy as np
from tqdm import tqdm

from detector import Detector, render_result
from utils import  xywh2xyxy, compute_IOU


def main():
    # det config
    config_path = '/home/liwang/projects/mmdetection/configs/lw/server/gfl_swin-s_fpn_1x_widerface_FDDB_4kface_scaleaware_640_256_256_CLR.py'
    checkpoint_path = '/home/liwang/work_dirs/face_det/server/gfl_swin-s_fpn_1x_widerface_FDDB_4kface_scaleaware_640_256_256_CLR/epoch_24.pth'
    min_score = 0.5

    detector = Detector(config_path, checkpoint_path, min_score)

    img_save_root = '/home/liwang/data/OxfordHand/OxfordHand/images'
    lmdb_root = '/home/liwang/data/OxfordHand/OxfordHand/lmdb'
    anno_root = '/home/liwang/data/OxfordHand/OxfordHand/face_anno'

    folder_names = ['test', 'train', 'validation']
    for sub in folder_names:
        lmdb_path = os.path.join(lmdb_root, sub, 'lmdb')
        img_save_dir = os.path.join(img_save_root, sub)
        os.makedirs(img_save_dir, exist_ok=True)
        
        anno_path = os.path.join(anno_root, sub+'.txt')
        
        f = open(anno_path, 'w')
        
        env_db = lmdb.open(lmdb_path, readonly=True, lock=False) 
        txn = env_db.begin()
        for key, value in tqdm(txn.cursor()):
            key = key.decode()
            if key.endswith('.jpg'):
                data = bytearray(value)
                img = cv2.imdecode(np.asarray(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                img_h = img.shape[0]
                img_w = img.shape[1]
                
                # detect
                result = detector.predict(img)
                render_result(img, result, detector.classes, detector.colors)
                
                
                # anno str
                img_anno_str = ' '
                for box in result:
                    assert len(box) == 6
                    img_anno_str += '{} {} {} {} {} '.format(box[2], box[3], box[4], box[5], 1)
                    # print(img_anno_str)

                img_anno = key + " {} {} {}".format(img_w, img_h, len(result)) + img_anno_str
                f.writelines(img_anno + '\n')
                
                img_save_path = os.path.join(img_save_dir, key)
                if not os.path.exists(img_save_path):
                    cv2.imwrite(img_save_path, img)
        env_db.close()
        f.close()


if __name__ == '__main__':
    main()