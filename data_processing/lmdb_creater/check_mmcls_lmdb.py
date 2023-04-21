import os
import cv2
import lmdb
import json
import numpy as np
from tqdm import tqdm
from glob import glob

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

# train_v1_lmdb_names = get_lmdb_names('/home/liwang/data/foot_det/shopee_shoes/train_v1/lmdb')
# train_v2_lmdb_names = get_lmdb_names('/home/liwang/data/foot_det/shopee_shoes/train_v2/lmdb')
img_shoes_with_leg_lmdb_names = get_lmdb_names('/home/liwang/data/foot_det/shopee_shoes/shoes_with_leg/lmdb')

# with open('train_v1_names.txt', 'w') as f:
#     for name in train_v1_lmdb_names:
#         f.write(name + '\n')
# with open('train_v2_names.txt', 'w') as f:
#     for name in train_v2_lmdb_names:
#         f.write(name + '\n')
with open('img_shoes_with_leg_lmdb_names.txt', 'w') as f:
    for name in img_shoes_with_leg_lmdb_names:
        f.write(name + '\n')

# name_list_v1 = os.listdir('/home/liwang/data/shopee_shoes/spu_comment/shoes_with_leg/shoes_with_leg_1')
# name_list_v2 = os.listdir('/home/liwang/data/shopee_shoes/spu_comment/shoes_with_leg/shoes_with_leg_2')

# for name in name_list_v2:
#     if name not in train_v2_lmdb_names:
#         if name in train_v1_lmdb_names:
#             print('{} in V1'.format(name))
#         else:
#             print('222 {}'.format(name))
#     else:
#         print('success 222')

# for name in name_list_v1:
#     if name not in train_v1_lmdb_names:
#         if name in train_v2_lmdb_names:
#             print('{} in V2'.format(name))
#         else:
#             print('111 {}'.format(name))
#     else:
#         print('success 111')