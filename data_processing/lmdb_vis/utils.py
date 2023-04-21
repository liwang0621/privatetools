'''
Author       : Li Wang
Date         : 2021-12-28 07:51:53
LastEditors  : wang.li
LastEditTime : 2022-07-04 16:53:32
FilePath     : /privatetools/data_processing/lmdb_vis/utils.py
Description  : 
'''
import cv2
from unicodedata import name
import numpy as np

def load_annotations(ann_file):
        annos = []
        with open(ann_file, "r") as f:
            for line in f.readlines():
                info = line.strip().split()
                img_name = info[0]
                img_w = int(info[1])
                img_h = int(info[2])
                obj_num = int(info[3])
                bboxes = []
                for obj_idx in range(0, obj_num):
                    x1 = float(info[4 + obj_idx * 5])
                    y1 = float(info[5 + obj_idx * 5])
                    x2 = float(info[6 + obj_idx * 5])
                    y2 = float(info[7 + obj_idx * 5])
                    label = int(info[8 + obj_idx * 5])
                    score = float(info[8 + obj_idx * 5])
                    if x2 <= x1 or y2 <= y1:
                        continue
                    bbox = [x1, y1, x2, y2]
                    bbox = [label, score] + bbox
                    bboxes.append(bbox)
                annos.append((img_name, bboxes))
        return annos


def load_annotations_label(ann_file):
        annos = []
        with open(ann_file, "r") as f:
            for line in f.readlines():
                info = line.strip().split()
                img_name = info[0]
                img_w = int(info[1])
                img_h = int(info[2])
                obj_num = int(info[3])
                bboxes = []
                for obj_idx in range(0, obj_num):
                    x1 = float(info[4 + obj_idx * 5])
                    y1 = float(info[5 + obj_idx * 5])
                    x2 = float(info[6 + obj_idx * 5])
                    y2 = float(info[7 + obj_idx * 5])
                    label = int(info[8 + obj_idx * 5])
                    if x2 <= x1 or y2 <= y1:
                        continue
                    bbox = [x1, y1, x2, y2]
                    bbox = [label] + bbox
                    bboxes.append(bbox)
                annos.append((img_name, bboxes))
        return annos

def draw_lmks(img, lmks):
    point_size = 2
    point_color = (0, 255, 255) # BGR
    thickness = 4 #  0 、4、8
    for lmk_id, coor in enumerate(lmks):
        print((int(coor[0]),int(coor[1])))
        cv2.circle(img, (int(coor[0]),int(coor[1])), point_size, point_color, thickness)
        render_text = "{}".format(lmk_id)
        cv2.putText(img, render_text, (int(coor[0]),int(coor[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img