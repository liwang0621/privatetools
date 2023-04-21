import os
import cv2
import json
import lmdb
import numpy as np
from glob import glob
import pandas as pd
from torch import ge
from tqdm import tqdm
from tqdm import tqdm

def get_lmdb_names(lmdb_path):
    env_db = lmdb.open(lmdb_path, readonly=True, lock=False) 
    txn = env_db.begin()
    names = []
    for key,value in tqdm(txn.cursor()):
        if key.decode() == 'num-samples':
            continue
    #     data = bytearray(value)
    #     img = cv2.imdecode(np.asarray(data, dtype=np.uint8),
    #                     cv2.IMREAD_COLOR)
    #     if img.shape[0] < 1 or img.shape[1] < 1:
    #         print(key, img.shape)
        names.append(key.decode())
    return names

def annostr2dict(annostr):
    info = annostr.strip().split()
    img_name = info[0]
    img_w = int(info[1])
    img_h = int(info[2])
    Label = int(info[3])

    out = {
        'img_name': img_name,
        'label': Label,
        'img_w': img_w,
        'img_h': img_h
    }
    return out

def load_annos(anno_path):
    annos = []
    with open(anno_path, "r") as f:
        for line in tqdm(f.readlines()):
            anno = annostr2dict(line)
            annos.append(anno)
    return annos


def save_annos(annos, save_anno_path):
    with open(save_anno_path, "w") as f:
        for anno in tqdm(annos):
            img_name = anno['img_name']
            img_w = anno['img_w']
            img_h = anno['img_h']
            label = anno['label']
            img_anno = img_name + " {} {} {}".format(img_w, img_h, label)

            f.writelines(img_anno + '\n')


label_1 = []
data = pd.read_excel('/home/liwang/人脚关键点.第二批有效22494-数据堂20220708.xlsx')
for name in data['有效文件名']:
    label_1.append(name)

# label_1 = label_1[:10]
label_1_num = len(label_1)


train_v1_lmdb_names = get_lmdb_names('/home/liwang/data/foot_det/shopee_shoes/train_v1/lmdb')
train_v2_lmdb_names = get_lmdb_names('/home/liwang/data/foot_det/shopee_shoes/train_v2/lmdb')

name_list = []
name_list.extend(os.listdir('/home/liwang/data/shopee_shoes/spu_comment/shoes_with_leg/shoes_with_leg_1'))
name_list.extend(os.listdir('/home/liwang/data/shopee_shoes/spu_comment/shoes_with_leg/shoes_with_leg_2'))

train_v1_names = []
train_v2_names = []
for name in name_list:
    if name in train_v1_lmdb_names:
        train_v1_names.append(name)
    elif name in train_v2_lmdb_names:
        train_v2_names.append(name)
    else:
        print('!!!!!!{}!!!!!'.format(name))


v1_label_0 = []
v1_label_1 = []
for name in train_v1_names:
    if name not in label_1:
        v1_label_0.append(name)
    else:
        v1_label_1.append(name)

v2_label_0 = []
v2_label_1 = []
for name in train_v2_names:
    if name not in label_1:
        v2_label_0.append(name)
    else:
        v2_label_1.append(name)

assert len(v1_label_1) + len(v2_label_1) == len(label_1)

v1_annos = []
env_db = lmdb.open('/home/liwang/data/foot_det/shopee_shoes/train_v1/lmdb', readonly=True, lock=False) 
txn = env_db.begin()
for name in tqdm(v1_label_0):
    data = txn.get(name.encode())
    data = bytearray(data)
    img = cv2.imdecode(np.asarray(data, dtype=np.uint8),
                    cv2.IMREAD_COLOR)
    img_h, img_w, _ = img.shape
    anno = {
        'img_name': name,
        'label': 0,
        'img_w': img_w,
        'img_h': img_h
    }
    v1_annos.append(anno)
for name in tqdm(v1_label_1):
    data = txn.get(name.encode())
    data = bytearray(data)
    img = cv2.imdecode(np.asarray(data, dtype=np.uint8),
                    cv2.IMREAD_COLOR)
    img_h, img_w, _ = img.shape
    anno = {
        'img_name': name,
        'label': 1,
        'img_w': img_w,
        'img_h': img_h
    }
    v1_annos.append(anno)


v2_annos = []
env_db = lmdb.open('/home/liwang/data/foot_det/shopee_shoes/train_v2/lmdb', readonly=True, lock=False) 
txn = env_db.begin()
for name in tqdm(v2_label_0):
    data = txn.get(name.encode())
    data = bytearray(data)
    img = cv2.imdecode(np.asarray(data, dtype=np.uint8),
                    cv2.IMREAD_COLOR)
    img_h, img_w, _ = img.shape
    anno = {
        'img_name': name,
        'label': 0,
        'img_w': img_w,
        'img_h': img_h
    }
    v2_annos.append(anno)
for name in tqdm(v2_label_1):
    img_path = os.path.join('/home/liwang/data/shopee_shoes/spu_comment/shoes_with_leg/shoes_with_leg_2', name)
    img = cv2.imread(img_path)
    img_h, img_w, _ = img.shape
    anno = {
        'img_name': name,
        'label': 1,
        'img_w': img_w,
        'img_h': img_h
    }
    v2_annos.append(anno)


# print(train_annos)
save_annos(v1_annos, '/home/liwang/data/foot_det/foot_cls/image_v1_anno_train.txt')

save_annos(v2_annos, '/home/liwang/data/foot_det/foot_cls/image_v2_anno_train.txt')




