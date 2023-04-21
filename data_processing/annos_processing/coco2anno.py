'''
Author       : Li Wang
Date         : 2022-04-21 14:52:17
LastEditors  : Li Wang
LastEditTime : 2022-04-21 18:09:19
FilePath     : /privatetools/data_processing/annos_processing/coco2anno.py
Description  : 
'''
import json
from tqdm import tqdm
from utils import save_annos

def xywh2xyxy(box):
    out = []
    out.append(box[0])
    out.append(box[1])
    out.append(box[0] + box[2])
    out.append(box[1] + box[3])
    return out

def coco2anno(cocojson_path, annotxt_save_path):
    data = json.load(open(cocojson_path, 'r'))

    # 首先得到数据的categories的关键字
    category = data['categories']
    category_id ={}
    for category_per in category:
        id = category_per['id']
        cls = category_per['name']
        category_id[id] = cls

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

    assert len(imageID_all_info) == len(anno_map)

    annos = []
    for img_id in anno_map:
        # print(img_id)
        # print(imageID_all_info[img_id])
        img_name = imageID_all_info[img_id]['filename']
        img_name = img_name.split(img_name.split('patch')[0])[-1]
        img_w = imageID_all_info[img_id]['width']
        img_h = imageID_all_info[img_id]['height']
        bboxes = []
        for bbox in anno_map[img_id]:
            category_id = bbox['category_id']
            box = bbox['bbox']
            box = xywh2xyxy(box)
            if bbox['iscrowd']:
                category_id = -1
            box.append(category_id)
            bboxes.append(box)
        item = {
            'img_name': img_name,
            'bboxes': bboxes,
            'img_w': img_w,
            'img_h': img_h,
        }
        annos.append(item)

    save_annos(annos, save_anno_path=annotxt_save_path)

if __name__=='__main__':
    cocojson_path = '/home/liwang/data/object365/tmp/zhiyuan_objv2_train.json'
    annotxt_save_path = '/home/liwang/data/general_det/object365/train/anno.txt'
    coco2anno(cocojson_path, annotxt_save_path)