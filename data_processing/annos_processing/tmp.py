import os
from tqdm import tqdm

from utils import save_annos, load_annos

train_v1_namelist = []
with open('/Users/wang.li/remote_projects/privatetools/data_processing/lmdb_creater/train_v1_names.txt', 'r') as f:
    for name in f.readlines():
        name = name.strip()
        train_v1_namelist.append(name)

train_v2_namelist = []
with open('/Users/wang.li/remote_projects/privatetools/data_processing/lmdb_creater/train_v2_names.txt', 'r') as f:
    for name in f.readlines():
        name = name.strip()
        train_v2_namelist.append(name)


img_shoes_with_leg_lmdb_namelist = []
with open('/Users/wang.li/remote_projects/privatetools/data_processing/lmdb_creater/img_shoes_with_leg_lmdb_names.txt', 'r') as f:
    for name in f.readlines():
        name = name.strip()
        img_shoes_with_leg_lmdb_namelist.append(name)


video_10w_shoes_with_leg_10w_lmdb_namelist = []
with open('/Users/wang.li/remote_projects/privatetools/data_processing/lmdb_creater/video_10w_shoes_with_leg_10w.txt', 'r') as f:
    for name in f.readlines():
        name = name.strip()
        video_10w_shoes_with_leg_10w_lmdb_namelist.append(name)



annos = load_annos('/Users/wang.li/remote_projects/privatetools/train_labeled_v9.txt')

img_shoes_with_leg_lmdb_annos = []
train_v2_annos = []
train_v1_annos = []
video_10w_shoes_with_leg_10w_annos = []

for anno in tqdm(annos):
    name = anno['img_name']
    if name in img_shoes_with_leg_lmdb_namelist:
        print('in img_shoes_with_leg_lmdb_namelist')
        img_shoes_with_leg_lmdb_annos.append(anno)
    elif name in train_v2_namelist:
        print('in train_v2_namelist')
        train_v2_annos.append(anno)
    elif name in train_v1_namelist:
        print('in train_v1_namelist')
        train_v1_annos.append(anno)
    elif name in video_10w_shoes_with_leg_10w_lmdb_namelist:
        print('in video_10w_shoes_with_leg_10w_lmdb_namelist')
        video_10w_shoes_with_leg_10w_annos.append(anno)
    else:
        print(name)

# from IPython import embed; embed()
if len(train_v1_annos) > 0:
    save_annos(train_v1_annos, 'train_v1_labeled_v9.txt')
if len(train_v2_annos) > 0:
    save_annos(train_v2_annos, 'train_v2_labeled_v9.txt')
if len(img_shoes_with_leg_lmdb_annos) > 0:
    save_annos(img_shoes_with_leg_lmdb_annos, 'img_shoes_with_leg_lmdb_labeled_v9.txt')
if len(video_10w_shoes_with_leg_10w_annos) > 0:
    save_annos(video_10w_shoes_with_leg_10w_annos, 'video_10w_shoes_with_leg_10w_labeled_v9.txt')


# source_dir= '/Users/wang.li/Downloads/人脚关键点交付2-8006张.数据堂/标注结果'

# for json_name in os.listdir(source_dir):
#     if json_name.endswith('.json'):
#         img_name = json_name.replace('.json', '.jpg')
#         if img_name in img_shoes_with_leg_lmdb_namelist:
#             cmd = 'cp {}/{} /Users/wang.li/Downloads/人脚关键点交付2-8006张.数据堂/img_shoes_with_leg_lmdb_res/'.format(source_dir, json_name)
#         elif img_name in train_v2_namelist:
#             cmd = 'cp {}/{} /Users/wang.li/Downloads/人脚关键点交付2-8006张.数据堂/train_v2_res/'.format(source_dir, json_name)
#         else:
#             print(json_name)
#             raise FileExistsError
#         print(cmd)
#         os.system(cmd)






