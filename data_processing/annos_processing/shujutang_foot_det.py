import os
import json
from re import A
import numpy as np
from glob import glob
from tqdm import tqdm

from utils import save_annos

def create_bbox_by_keypoints(keypoints):
    x1 = np.min(keypoints[:, 0])
    y1 = np.min(keypoints[:, 1])
    x2 = np.max(keypoints[:, 0])
    y2 = np.max(keypoints[:, 1])
    return [x1, y1, x2, y2]


# def get_lmks_from_json(json_path):
#     data = json.load(open(json_path, 'r'))




if __name__ == "__main__":
    annos = []
    for json_path in tqdm(glob('/Users/wang.li/Downloads/人脚关键点交付9-4693张.数据堂/标注结果/*.json')):
        data = json.load(open(json_path, 'r'))
        img_name = data['filePath']
        # if img_name == '06921a15ca63efdb9185cd352e704bdf.jpg':
        #     from IPython import embed; embed()
        img_w = data['info']['width']
        img_h = data['info']['height']

        # foot_num = len()
        lmks = {}
        masks = []
        for it in data['dataList']:
            if it['shapeType'] == 'point':
                # print(it)
                objectId = it['properties']['objectId']
                if objectId not in lmks:
                    lmks[objectId] = {}
                point_id = it['id']
                point = it['coordinates']
                lmks[objectId][point_id] = point
            elif it['shapeType'] == 'rectangle':
                box = it['coordinates']
                box = [box[0][0], box[0][1], box[1][0], box[1][1]]
                masks.append(box)
        lmk_boxes = []
        for key in lmks:
            lmk = []
            try:
                assert len(lmks[key]) == 9
            except:
                print(json_path)
                continue
            for i in range(1,10):
                point = lmks[key][i]
                lmk.append(point)
            box = create_bbox_by_keypoints(np.array(lmk))
            lmk_boxes.append(box)

        bboxes = []
        for box in lmk_boxes:
            box.append(1)
            bboxes.append(box)
        for box in masks:
            box.append(0)
            bboxes.append(box)  
        anno = {
            'img_name': img_name,
            'bboxes': bboxes,
            'img_w': img_w,
            'img_h': img_h
        }
        annos.append(anno)

save_annos(annos, 'train_labeled_v9.txt')