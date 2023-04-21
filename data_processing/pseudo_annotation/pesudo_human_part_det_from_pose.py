'''
Author       : Li Wang
Date         : 2022-04-11 18:14:31
LastEditors  : Li Wang
LastEditTime : 2022-04-12 11:18:42
FilePath     : /privatetools/data_processing/pseudo_annotation/pesudo_human_part_det_from_pose.py
Description  : 
'''
import os
import cv2
import sys
import lmdb
import numpy as np
from tqdm import tqdm

from pose_estimation import PoseEstimation

sys.path.append('..')
from annos_processing.utils import load_annos, save_annos

def create_bbox_by_keypoints(keypoints, label, enlarge_ratio=1.3):
    if label == 1:
        min_short_long_ratio = 0.6
    else:
        min_short_long_ratio = 0.35
    if np.sum(keypoints[:, 2] > 0) != keypoints.shape[0]:
        return None
    ori_x1 = np.min(keypoints[:, 0])
    ori_y1 = np.min(keypoints[:, 1])
    ori_x2 = np.max(keypoints[:, 0])
    ori_y2 = np.max(keypoints[:, 1])
    ctx = (ori_x1 + ori_x2) * 0.5
    cty = (ori_y1 + ori_y2) * 0.5
    ori_w = abs(ori_x2 - ori_x1)
    ori_h = abs(ori_y2 - ori_y1)
    min_size = max(ori_w, ori_h) * min_short_long_ratio
    ori_w = max(ori_w, min_size)
    ori_h = max(ori_h, min_size)

    enlarge_w = ori_w * enlarge_ratio
    enlarge_h = ori_h * enlarge_ratio
    x1 = ctx - enlarge_w * 0.5
    y1 = cty - enlarge_h * 0.5
    x2 = ctx + enlarge_w * 0.5
    y2 = cty + enlarge_h * 0.5
    return np.asarray([x1, y1, x2, y2])

def render(r_img, img_anns):
    for img_ann in img_anns:
        bbox, label = img_ann
        r_bbox = bbox.astype(np.int32)
        if label == -1:
            color = (0, 0, 255)
        elif label == 1:
            color = (0, 255, 0)
        else:
            color = (255, 255, 0)
        cv2.rectangle(r_img, (r_bbox[0], r_bbox[1]), (r_bbox[2], r_bbox[3]), color, 2)
    return r_img

if __name__=='__main__':
    config_path = '/home/liwang/projects/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vipnas_res50_coco_256x192.py'
    checkpoint_path = 'https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_res50_coco_256x192-cc43b466_20210624.pth'
    # 0: nose
    # 1: left_eye
    # 2: right_eye
    # 3: left_ear
    # 4: right_ear
    # 5: left_shoulder
    # 6: right_shoulder
    # 7: left_elbow
    # 8: right_elbow
    # 9: left_wrist
    # 10: right_wrist
    # 11: left_hip
    # 12: right_hip
    # 13: left_knee
    # 14: right_knee
    # 15: left_ankle
    # 16: right_ankle

    pose_estimation = PoseEstimation(config_path, checkpoint_path, bbox_thr=0.5, kpt_thr=0.5)

    name_list_path = '/home/liwang/data/human_part_det/yydata/anno_hand_7w.txt'
    name_list = load_annos(name_list_path)

    lmdb_path = '/home/liwang/data/yy-data/lmdb/lmdb'
    env_db = lmdb.open(lmdb_path, readonly=True, lock=False) 
    txn = env_db.begin()

    # data = txn.get(img_name.encode())
    # data = bytearray(data)
    # img = cv2.imdecode(np.asarray(data, dtype=np.uint8),
    #                 cv2.IMREAD_COLOR)

    i = 0
    annos = []
    for it in tqdm(name_list[30000:]):
        img_name = it['img_name']
        # img = cv2.imread('input.png')

        data = txn.get(img_name.encode())
        data = bytearray(data)
        img = cv2.imdecode(np.asarray(data, dtype=np.uint8),
                        cv2.IMREAD_COLOR)

        img_h = img.shape[0]
        img_w = img.shape[1]

        anno = {
            'img_name': img_name,
            'img_w': img_w,
            'img_h': img_h
        }


        pose_results = pose_estimation.predict(img)

        upper_label=1
        lower_label=2
        cvt_anns = [] = []
        bboxes = []
        tag = True
        for pose_result in pose_results:
            keypoints = pose_result['keypoints']
            upper_keypoints = keypoints[[5, 6, 11, 12], :]
            lower_keypoints = keypoints[[11, 12, 13, 14, 15, 16], :]
            upper_bbox = create_bbox_by_keypoints(upper_keypoints, upper_label)
            lower_bbox = create_bbox_by_keypoints(lower_keypoints, lower_label)
            if upper_bbox is not None and lower_bbox is not None:
                if upper_bbox[1] > lower_bbox[1]:
                    tag = False

            if upper_bbox is not None:
                cvt_anns.append((upper_bbox, upper_label))
                bboxes.append(list(upper_bbox) + [upper_label])
            if lower_bbox is not None:
                cvt_anns.append((lower_bbox, lower_label))
                bboxes.append(list(lower_bbox) + [lower_label])
        # cv2.imwrite('demo_label_{}.png'.format(i), render(img, cvt_anns))
        i = i+1
        anno['bboxes'] = bboxes
        if tag:
            annos.append(anno)
        #     cv2.imwrite('demo_label_{}.jpg'.format(i), render(img, cvt_anns))
        # else:
        #     cv2.imwrite('demo_no_{}.jpg'.format(i), render(img, cvt_anns))
    # from pdb import set_trace; set_trace()
    save_annos(annos, 'demo_40000.txt')
    
