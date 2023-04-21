import os
import cv2
import numpy as np
from tqdm import tqdm
from utils import save_annos, get_filelist, create_bbox_by_keypoints


def get_lmks_from_txt(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    img_name = lines[0].strip()
    point_num = int(lines[1].strip())

    lmks = []
    for i in range(point_num):
        x = float(lines[i * 3 + 2].strip())
        y = float(lines[i * 3 + 3].strip())
        tag = int(lines[i * 3 + 4].strip())
        lmks.append([x,y, tag])
    return lmks

root = '/home/liwang/data/face_mask/'
Filelist = []
Filelist = get_filelist(root, Filelist)

annos = []
label = 1
for img_path in tqdm(Filelist):
    txt_path = img_path.replace('.jpg', '.txt')
    lmks = get_lmks_from_txt(txt_path)
    mask_ratio = (251 - np.sum(np.array(lmks)[:,2]))/ 251
    if mask_ratio > 0.35:
        box = create_bbox_by_keypoints(np.array(lmks))
        box.append(label)

        img = cv2.imread(img_path)
        img_name = img_path.split(root)[-1]
        img_name = img_name.replace('/', '_')
        img_name = img_name.replace(' ', '')


        img_h = img.shape[0]
        img_w = img.shape[1]

        bboxes = [box]

        anno = {
            'img_name': img_name,
            'bboxes': bboxes,
            'img_w': img_w,
            'img_h': img_h
        }
        # print(anno)
        annos.append(anno)
    # print('{} : {}'.format(mask_ratio, img_path))

print('anno num: {}'.format(len(annos)))
save_annos(annos, '/home/liwang/data/face_mask/anno_0.35.txt')