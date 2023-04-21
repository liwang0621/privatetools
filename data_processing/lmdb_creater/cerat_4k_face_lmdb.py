import numpy as np
import lmdb
import math
import os
import cv2
import json
import argparse
from tqdm import tqdm
from glob import glob
from loguru import logger
from IPython import embed

batch_size = 100


def xywh2xyxy(box):
    out = []
    out.append(box[0])
    out.append(box[1])
    out.append(box[0] + box[2])
    out.append(box[1] + box[3])
    return out


def load_img_names(label_file_path):
    image_list = []
    with open(label_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            its = json.loads(line)
    its = its['items']
    
    for it in its:
        meta = {}
        try:
            meta['path'] = it['filename']
        except:
            embed()
        meta['width'] = it['width']
        meta['height'] = it['height']
        boxes_list = []
        for gt in it['gtboxes']:
            if gt['tag'] == 'Face':
                if gt['extra']['ignore'] == 1:
                    label = 0
                elif gt['extra']['ignore'] == 0:
                    label = 1
                else:
                    logger.info('data ignore wrong!!!!!')
                    break
                box = xywh2xyxy(gt['box'])
                box.append(label)
                boxes_list.append(box)
            else:
                logger.info('data tag wrong!!!!!')
                break
        meta['boxes'] = boxes_list
        image_list.append(meta)
    return image_list


def create(img_names, img_dir, save_root):
    save_anno_path = os.path.join(save_root, "anno.txt")
    save_lmdb_dir = os.path.join(save_root, "lmdb")
    if not os.path.isdir(save_lmdb_dir):
        os.makedirs(save_lmdb_dir)

    env = lmdb.open(save_lmdb_dir, map_size=int(1e12))
    txn = env.begin(write=True)
    with open(save_anno_path, "w") as f:
        for img_idx in tqdm(range(0, len(img_names))):
            # img_name, img_anno = img_names[img_idx]
            img_name = img_names[img_idx]['path']
            img_anno = img_names[img_idx]['boxes']
            img_path = os.path.join(img_dir, img_name)
            if ' ' in img_name:
                logger.info('there is a space in the path')
                img_name = img_name.replace(' ', '_')
            img = cv2.imread(img_path)
            if img is None:
                logger.info('there is a problom!!!!!!!')
                continue
            img_data = cv2.imencode(".jpg", img)[1].tobytes()
            txn.put(img_name.encode(), img_data)
            img_h = img.shape[0]
            img_w = img.shape[1]

            
            img_anno_str = ' '
            for box in img_anno:
                assert len(box) == 5
                img_anno_str += '{} {} {} {} {} '.format(box[0], box[1], box[2], box[3], box[4])
                # print(img_anno_str)

            img_anno = img_name + " {} {} {}".format(img_w, img_h, len(img_anno)) + img_anno_str

            f.writelines(img_anno + '\n')
    txn.commit()
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_dir", default='/home/liwang/data/4K-Face')
    parser.add_argument("--img_dir", default='/home/liwang/data/4K-Face/images')
    parser.add_argument("--save_root", default='/home/liwang/data/face_det/4K-Face')

    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)

    anno_path_list = glob(args.anno_dir + '/*.json')

    image_list = []
    for anno_path in anno_path_list:
        image_list.extend(load_img_names(anno_path))

    from IPython import embed; embed()

    # img_names = load_img_names(args.anno_path)
    create(image_list, args.img_dir, args.save_root)
