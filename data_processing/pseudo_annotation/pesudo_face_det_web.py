'''
Author       : Li Wang
Date         : 2021-12-22 03:58:38
LastEditors  : wang.li
LastEditTime : 2022-07-13 15:23:39
FilePath     : /privatetools/data_processing/pseudo_annotation/pesudo_face_det_web.py
Description  : 
'''
import os
import cv2
import lmdb
import requests
import numpy as np
from tqdm import tqdm

from detector import Detector, render_result
from utils import  xywh2xyxy, compute_IOU

requests.adapters.DEFAULT_RETRIES = 5

def get_img_from_web(url):
    data = requests.get(url, headers={'Connection':'close'})
    data = data.content
    data = bytearray(data)
    img = cv2.imdecode(np.asarray(data, dtype=np.uint8),
                    cv2.IMREAD_COLOR)
    return img


def main():
    # det config
    config_path = '/home/liwang/projects/mmdetection/configs/lw/server/gfl_swin-s_fpn_1x_widerface_FDDB_4kface_scaleaware_640_256_256_CLR.py'
    checkpoint_path = '/home/liwang/work_dirs/face_det/server/gfl_swin-s_fpn_1x_widerface_FDDB_4kface_scaleaware_640_256_256_CLR/epoch_24.pth'
    min_score = 0.3

    detector = Detector(config_path, checkpoint_path, min_score, device='cuda')

    url_root = 'https://spx.shopee.sg/spx-perm-202201/'
    url_path = '/home/liwang/data/gongyinglian/photos_100000.txt'

    anno_save_path = '/home/liwang/data/gongyinglian/anno_photos_100000.txt'
    lmdb_save_dir = '/home/liwang/data/gongyinglian/photos_100000/lmdb'
    # if not os.path.exists(lmdb_save_dir):
    os.makedirs(lmdb_save_dir, exist_ok=True)

    demo_img_save_dir = '/home/liwang/data/gongyinglian/photos_100000/images'
    os.makedirs(demo_img_save_dir, exist_ok=True)
    demo_num = 100
    

    f = open(anno_save_path, 'w')
    env = lmdb.open(lmdb_save_dir, map_size=int(1e12))
    txn = env.begin(write=True)

    with open(url_path, 'r') as ff:
        for line in tqdm(ff.readlines()):
            line = line.strip()
            url = url_root + line
            img = get_img_from_web(url)

            img_h = img.shape[0]
            img_w = img.shape[1]
            
            # detect
            result = detector.predict(img)
            if len(result) > 0:
                img_data = cv2.imencode(".jpg", img)[1].tobytes()
                txn.put(line.encode(), img_data)
                
                # anno str
                img_anno_str = ' '
                for box in result:
                    assert len(box) == 6
                    img_anno_str += '{} {} {} {} {} '.format(box[2], box[3], box[4], box[5], 1)
                    # print(img_anno_str)

                img_anno = line + " {} {} {}".format(img_w, img_h, len(result)) + img_anno_str
                f.writelines(img_anno + '\n')
                if demo_num > 0:
                    render_result(img, result, detector.classes, detector.colors)
                    img_save_path = os.path.join(demo_img_save_dir, os.path.basename(line))
                    cv2.imwrite(img_save_path, img)
                    demo_num-=1

    f.close()
    txn.commit()
    env.close()

if __name__ == '__main__':
    main()