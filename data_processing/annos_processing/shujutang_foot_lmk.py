import json
import os
from unittest import result
import cv2
import shutil
import numpy as np
import multiprocessing
from re import L, M
from tqdm import tqdm
from multiprocessing import Manager

def xyxy2xywh(box):
    # input box: [x,y,w,h]
    new_box = []
    new_box.append(box[0])
    new_box.append(box[1])
    new_box.append(box[2] - box[0])
    new_box.append(box[3] - box[1])
    return new_box

def create_bbox_by_keypoints(keypoints):
    x1 = np.min(keypoints[:, 0])
    y1 = np.min(keypoints[:, 1])
    x2 = np.max(keypoints[:, 0])
    y2 = np.max(keypoints[:, 1])
    return [x1, y1, x2, y2]

def expand_bbox(box, image_w, image_h, expand_ratio=0):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    ## get the center and the radius
    cx = x1 + w//2
    cy = y1 + h//2
    x1 = max(0, cx - w//2 * (1 + expand_ratio))
    y1 = max(0, cy - h//2 * (1 + expand_ratio))
    x2 = min(cx + w//2 * (1 + expand_ratio), image_w)
    y2 = min(cy + h//2 * (1 + expand_ratio), image_h)
    
    return (int(x1), int(y1), int(x2), int(y2))

def load_jsons(json_dir):
    image_path = ''
    results = []
    files = os.listdir(json_dir)
    multithread = 32
    nr_lines = len(files)
    nr_liner_per_worker = np.ceil(nr_lines / multithread)
    manager = Manager()
    # return_list = manager.list() 也可以使用列表list
    return_dict = manager.dict()
    jobs = []
    for i in range(multithread):
        start = int(i * nr_liner_per_worker)
        end = int(min(start + nr_liner_per_worker, nr_lines))
        image_worker = files[start:end]

        p = multiprocessing.Process(target=worker_image,
                                    args=(i, image_worker, json_dir, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    
    result_dict = dict(return_dict)
    for proc_num in result_dict:
        results += result_dict[proc_num]
    
    return results

def worker_image(procnum, images_worker, json_dir, return_dict):
    image_path = '/home/liwang/data/foot_lmk/spu_shoes/image_v2_labeled_v1/images'
    out = []
    for file in tqdm(images_worker):
        try:
            data = json.load(open(os.path.join(json_dir, file), 'r'))
        except:
            print('73737373737 {}'.format(file))
        img_name = data['filePath']
        img_w = data['info']['width']
        img_h = data['info']['height']

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
        keypoints = []
        for key in lmks:
            lmk = []
            assert len(lmks[key]) == 9
            for i in range(1,10):
                point = lmks[key][i]
                lmk.append(point)
            keypoints.append(lmk)
            box = create_bbox_by_keypoints(np.array(lmk))
            box = expand_bbox(box, img_w, img_h, expand_ratio=0.)
            lmk_boxes.append(box)

        res = {
            "image_name": img_name,
            "image_h": img_h,
            "image_w": img_w,
            "image_path": os.path.join(image_path, img_name),
            'bboxes': lmk_boxes,
            'keypoints': keypoints,
            'masks': masks
        }
        out.append(res)
    return_dict[procnum] = out

def add_info(json_text):
    json_text["info"] = {
        "description": "CrawlData",
        "version": "1.0",
        "year": "2022",
        "date_created": "2022/07/07"
    }
    json_text["licenses"] = ""
    
def add_categories(json_text):
    json_text["categories"] = [{
            "supercategory": "foot",
            "id": 1,
            "name": "foot",
            "keypoints": [
                'heel_bottom', # 脚后跟下面
                'foot_front', # 脚最前方
                'ankle_back', # 脚踝后面
                'feed_outside', # 脚掌外侧
                'feed_inside', #脚掌内侧
                'heel_outside', # 脚后跟外侧
                'heel_inside', # 脚后跟内侧
                'ankle_front', # 脚踝前面
                'toe_middle', # 脚第三趾根
            ],
            "skeleton": [
                [1,3],
                [3,8],
                [6,8],
                [7,8],
                [8,9],
                [2,9],
                [2,4],
                [2,5],
            ]
        }]

def add_annotations(json_text, results):
    anno_images = []
    anno_annotations = []
    anno_id = 0
    for idx, meta in enumerate(results):
        img_id =  idx + 1
        img_info = {
            "file_name": meta['image_name'],
            "height": meta['image_h'],
            "width": meta['image_w'],
            "id": img_id
        }
        anno_images.append(img_info)

        for box, lmk in zip(meta['bboxes'], meta['keypoints']):
            box = xyxy2xywh(box)
            lmk_coco = []
            for point in lmk:
                point += [2]
                lmk_coco += point
            anno = {
                'iscrowd': 0,
                'keypoints': lmk_coco,
                'bbox': box,
                'category_id': 1,
                'id': anno_id,
                'image_id': img_id
            }
            anno_annotations.append(anno)
            anno_id += 1
        
        for box in meta['masks']:
            box = xyxy2xywh(box)
            anno = {
                'iscrowd': 1,
                'keypoints': [0] * 27,
                'bbox': box,
                'category_id': 1,
                'id': anno_id,
                'image_id': img_id
            }
            anno_annotations.append(anno)
            anno_id += 1
    json_text['images'] = anno_images
    json_text['annotations'] = anno_annotations

def json_pipeline(image_path, save_path, data_type):
    json_text = {}
    print('writing info')
    add_info(json_text)
    print('loading images')
    results = load_jsons(image_path)
    print('writing annotations')
    add_annotations(json_text, results)
    # print('writing images')
    # add_images(json_text, results, idxs, data_type)
    print('writing categories')
    add_categories(json_text)
    f = open(save_path, 'w')
    json_data = json.dumps(json_text, indent=4, separators=(',', ': '))
    f.write(json_data)
    f.close()


def unittest():
    data_type = 'train'
    image_path = "/Users/wang.li/Downloads/人脚关键点交付9-4693张.数据堂/标注结果"
    output_path = f"sjt_labeled_v9_exp0_{data_type}.json"
    json_pipeline(image_path, output_path, data_type)


if __name__ == '__main__':
    unittest()