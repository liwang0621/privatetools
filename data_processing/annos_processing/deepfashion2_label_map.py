import os
import cv2
import json
from glob import glob
from IPython import embed
from tqdm import tqdm
from loguru import logger

from utils import load_annos, save_annos


def f_label_map(annos, mapper):
    out = []
    for anno in annos:
        img_w = anno['img_w']
        img_h = anno['img_h']
        img_name = anno['img_name']
        bboxes = anno['bboxes']
        new_bboxes = []
        for box in bboxes:
            label = box[4]
            if label in mapper:
                box[4] = mapper[label]
                new_bboxes.append(box)
        out.append({
            'img_name': img_name,
            'bboxes': new_bboxes,
            'img_w': img_w,
            'img_h': img_h
        })
    return out

def json2anno(anno, img_name, img_w, img_h):
    out = {
        'img_name': img_name,
        'img_w': img_w,
        'img_h': img_h
    }
    bboxes = []
    nums = len(anno) - 2
    for i in range(nums):
        key = 'item{}'.format(i + 1)
        category_id = anno[key]['category_id']
        box = anno[key]['bounding_box']
        box.append(category_id)
        bboxes.append(box)
    out['bboxes'] = bboxes
    return out


label2id_map = {'人': 1,
 '动物': 2,
 '食物': 3,
 '服饰': 4,
 '交通工具': 5,
 '乐器': 6,
 '运动器械': 7,
 '电子产品': 8,
 '办公用品': 9,
 '家具家电': 10,
 '厨具': 11,
 '浴室用品': 12,
 '首饰': 13,
 '化妆品': 14,
 '玩具': 15,
 'ignore': -1}

deepfashion2_labels = [4, 8, 9, 12, 1, 7, 10, 2, 13, 11, 5, 6, 3]
label_id_map = {}
for label in deepfashion2_labels:
    label_id_map[label] = 4

logger.info('json 2 anno.txt')
annos = []
for json_path in tqdm(glob('/home/liwang/data/deepfashion2/train/annos/*')):
    img_name = os.path.basename(json_path).replace('json', 'jpg')
    img_path = os.path.join('/home/liwang/data/deepfashion2/train/image', img_name)
    try:
        img = cv2.imread(img_path)
        img_h = img.shape[0]
        img_w = img.shape[1]
    except:
        embed()
    anno = json.load(open(json_path, 'r'))
    anno = json2anno(anno, img_name, img_w, img_h)
    annos.append(anno)

annos = f_label_map(annos, label_id_map)
save_annos(annos, '/home/liwang/data/general_det/deepfashion2/train/anno_15class.txt')