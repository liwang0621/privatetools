import cv2
import os
import json
import numpy as np
from tqdm import tqdm
from utils import save_annos

def xywh2xyxy(box):
    out = []
    out.append(box[0])
    out.append(box[1])
    out.append(box[0] + box[2])
    out.append(box[1] + box[3])
    return out

def create_bbox_by_keypoints(keypoints, label=1, enlarge_ratio=1.35):
    min_short_long_ratio = 0.35

#     if np.sum(keypoints[:, 2] > 0) != keypoints.shape[0]:
#         return None
    # keypoints = keypoints[keypoints[:, 2] > 0]
    if keypoints.shape[0] == 0:
        return None
    try:
        ori_x1 = np.min(keypoints[:, 0])
    except:
        from pdb import set_trace;set_trace()

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
    return [x1, y1, x2, y2]

json_path = '/Users/wang.li/data/coco/foot_lmk/person_keypoints_val2017_foot_v1.json'
data = json.load(open(json_path, 'r'))
annotxt_save_path = '/Users/wang.li/data/coco/foot_lmk/foot_val1.txt'

# img_root = '/Users/wang.li/data/coco/val2017'

imageID_all_info = {}
for data_per in tqdm(data['images']):                        
    # 获取图像名字
    image_name = data_per['file_name']

    # 获取图像的像素和ID
    image_width = data_per['width']
    image_height = data_per['height']
    image_id = data_per['id']
    imageID_all_info[image_id]={'width':image_width,'height':image_height,'filename':image_name}

anno_map = {}
for data_per in tqdm(data['annotations']): 
    img_id = data_per['image_id']
    if img_id not in anno_map:
        anno_map[img_id] = [data_per]
    else:
        anno_map[img_id].append(data_per)

assert len(imageID_all_info) >= len(anno_map)

annos = []
nums = 0
for img_id in tqdm(imageID_all_info):
    img_name = imageID_all_info[img_id]['filename']
    img_w = imageID_all_info[img_id]['width']
    img_h = imageID_all_info[img_id]['height']
    bboxes = []
    for bbox in anno_map.get(img_id, []):
        category_id = bbox['category_id']
        assert category_id == 1
        box = bbox['bbox']
        box = xywh2xyxy(box)

        lmk = bbox['keypoints']
        keypoints = np.array(lmk).reshape(-1, 3)[-6:]
        left_lmk = keypoints[:3]
        right_lmk = keypoints[3:]

        if np.sum(left_lmk[:, 2] >0) < 3 or np.sum(right_lmk[:, 2] >0) < 3:
            ignore_box = box
            ignore_box.append(0)
            bboxes.append(box)

        if np.sum(left_lmk[:, 2] >0) == 3:
            box = create_bbox_by_keypoints(left_lmk)
            box.append(1)
            bboxes.append(box)
            nums +=1

        if np.sum(right_lmk[:, 2] >0) == 3:
            box = create_bbox_by_keypoints(right_lmk)
            box.append(1)
            bboxes.append(box)
            nums +=1

    item = {
        'img_name': img_name,
        'bboxes': bboxes,
        'img_w': img_w,
        'img_h': img_h,
    }
    if len(bboxes) > 0:
        annos.append(item)

print(nums)

save_annos(annos, save_anno_path=annotxt_save_path)