'''
Author       : Li Wang
Date         : 2021-12-22 03:58:38
LastEditors  : wang.li
LastEditTime : 2022-08-01 17:28:55
FilePath     : /privatetools/data_processing/pseudo_annotation/pesudo_face_det_file.py
Description  : 
'''
import os
import cv2
import lmdb
import sys
import requests
import numpy as np
from glob import glob
from tqdm import tqdm

from detector import Detector, render_result
from utils import  xywh2xyxy, compute_IOU

sys.path.append('../')
from annos_processing.utils import load_annos, save_annos

import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

requests.adapters.DEFAULT_RETRIES = 5

def draw_bboxes(img, bboxes):
    for bbox in bboxes:
        x1, y1, x2, y2, label = bbox
        color = [255,255,0]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        y_text = y1 + 20
        render_text = "{}".format(label)
        cv2.putText(img, render_text, (int(x1), int(y1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    return img


def get_img_from_web(url):
    data = requests.get(url, headers={'Connection':'close'})
    data = data.content
    data = bytearray(data)
    img = cv2.imdecode(np.asarray(data, dtype=np.uint8),
                    cv2.IMREAD_COLOR)
    return img

def get_img_name(img_path):
    # name = img_path.split('/home/liwang/data/face_raw_data/')[-1]
    name = os.path.basename(img_path)
    name = name.replace('/', '_')
    name = name.replace(' ', '')
    return name


def main():
    # # det config
    # config_path = '/home/liwang/projects/mmdetection/configs/lw/server/gfl_swin-s_fpn_1x_widerface_FDDB_4kface_scaleaware_640_256_256_CLR.py'
    # checkpoint_path = '/home/liwang/work_dirs/face_det/server/gfl_swin-s_fpn_1x_widerface_FDDB_4kface_scaleaware_640_256_256_CLR/epoch_24.pth'
    # min_score = 0.4

    # foot det config
    config_path = '/home/liwang/projects/mmdetection/configs/lw/foot_det/gfl_swin-s_fpn_1x_coco_sjt_v1v2v3v4x2_480_256_256.py'
    checkpoint_path = '/home/liwang/work_dirs/foot_det/gfl_swin-s_fpn_1x_coco_sjt_v1v2v3v4x2_640_256_256/epoch_24.pth'
    min_score = 0.4

    use_mp = False

    if use_mp:
        detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    else:
        detector = Detector(config_path, checkpoint_path, min_score, device='cuda')

    anno_save_path = '/home/liwang/data/foot_det/shopee_shoes_video_10w_other/anno.txt'

    demo_img_save_dir = '/home/liwang/data/foot_det/shopee_shoes_video_10w_other/demo'
    os.makedirs(demo_img_save_dir, exist_ok=True)
    demo_num = 1000

    wrong_det_img_save_dir = '/home/liwang/data/foot_det/shopee_shoes_video_10w_other/wrong_det_img'
    os.makedirs(wrong_det_img_save_dir, exist_ok=True)
    
    img_list= glob('/home/liwang/data/shopee_shoes/spu_comment/video_10w_cls_res/shoes_with_leg/*')
    print('img num: {}'.format(len(img_list)))

    wrong_det_num = 0

    annos = []
    label = 1
    demo_id = 0
    for img_path in tqdm(img_list):
        try:
            img = cv2.imread(img_path)

            img_name = get_img_name(img_path)

            img_h = img.shape[0]
            img_w = img.shape[1]
        except:
            continue
            
        # detect
        if use_mp:
            results = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            bboxes = []
            if not results.detections:
                print('!! {}'.format(img_path))
                wrong_det_num+=1
                wrong_det_img = draw_bboxes(img.copy(), bboxes)
                wrong_det_img_save_path = os.path.join(wrong_det_img_save_dir, img_name)
                cv2.imwrite(wrong_det_img_save_path, wrong_det_img)
            else:
                for detection in results.detections:
                    if not detection.location_data:
                        print('shahshshsshsh')
                    else:
                        relative_bounding_box = detection.location_data.relative_bounding_box
                        x1 = relative_bounding_box.xmin * img_w
                        y1 = relative_bounding_box.ymin * img_h
                        w = relative_bounding_box.width * img_w
                        h = relative_bounding_box.height * img_h
                        box = [x1, y1, x1+w, y1+h]
                        box.append(label)
                        bboxes.append(box)
                        anno = {
                                'img_name': img_name,
                                'bboxes': bboxes,
                                'img_w': img_w,
                                'img_h': img_h
                            }
                        annos.append(anno)
        else:
            result = detector.predict(img)
            bboxes = []
            if len(result) >= 1:
                # print(img_path)
                for it in result:
                    box = [it[2], it[3], it[4], it[5]]
                    box.append(label)
                    bboxes.append(box)
                anno = {
                        'img_name': img_name,
                        'bboxes': bboxes,
                        'img_w': img_w,
                        'img_h': img_h
                    }
                annos.append(anno)
            else:
                print('!! {}'.format(img_path))
                wrong_det_num+=1
                wrong_det_img = draw_bboxes(img.copy(), bboxes)
                wrong_det_img_save_path = os.path.join(wrong_det_img_save_dir, img_name)
                cv2.imwrite(wrong_det_img_save_path, wrong_det_img)
        
        if demo_id < demo_num:
            if len(bboxes) > 0:
                demo_img = draw_bboxes(img.copy(), bboxes)
                demo_img_save_path = os.path.join(demo_img_save_dir, img_name)
                try:
                    cv2.imwrite(demo_img_save_path, demo_img)
                except:
                    print()
                demo_id += 1

    save_annos(annos, anno_save_path)
    print('img num: {}'.format(len(img_list)))
    print('wrong ing num: {}'.format(wrong_det_num))



if __name__ == '__main__':
    main()