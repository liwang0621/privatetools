import os
from tkinter import image_names
import cv2
import sys
import lmdb
import numpy as np
from glob import glob
from tqdm import tqdm
from loguru import logger
from redis import DataError

sys.path.append('../')
from annos_processing.utils import get_filelist, load_annos

class LMDBCreater():
    def __init__(self, img_list:list, img_name_list:list, save_lmdb_path:str) -> None:
        assert len(img_list) == len(img_name_list)

        self.img_num = len(img_list)
        self.img_list = img_list
        self.img_name_list = img_name_list

        self.check_img_names()
        logger.info('Name verification succeeded')

        self.save_lmdb_path = save_lmdb_path
        if not os.path.exists(self.save_lmdb_path):
            os.makedirs(self.save_lmdb_path, exist_ok=True)
            logger.info('save_lmdb_path created Successfully')

        
        logger.info('There are {} images to be processed'.format(self.img_num))

    def check_img_names(self):
        for name in self.img_name_list:
            if ' ' in name:
                raise DataError('space in name')
            if '/' in name:
                raise DataError('/ in name')

    def run(self) -> None:
        env = lmdb.open(self.save_lmdb_path, map_size=int(1e12))
        txn = env.begin(write=True)

        for img_id in tqdm(range(self.img_num)):
            img_path = self.img_list[img_id]
            img_name = self.img_name_list[img_id]
            try:
                img = cv2.imread(img_path)
                img_h = img.shape[0]
                img_w = img.shape[1]
                if img is None:
                    logger.info('there is a problom!!!!!!!')
                    continue
            except:
                continue
            img_data = cv2.imencode(".jpg", img)[1].tobytes()
            txn.put(img_name.encode(), img_data)

        txn.commit()
        env.close()
        logger.info('success write lmdb in {}'.format(self.save_lmdb_path))


def get_img_name(img_path):
    name = img_path.split('/home/liwang/data/face_raw_data/')[-1]
    name = name.replace('/', '_')
    name = name.replace(' ', '')
    return name

if __name__ == "__main__":

    input_img_list = []
    input_img_name_list = []
    lmdb_save_path = '/home/liwang/data/face_det/dahuzi/lmdb'

    # annos = load_annos('/home/liwang/data/face_det/dachun/anno_0.35.txt')
    # annos_names = []
    # for it in annos:
    #     name = it['img_name']
    #     annos_names.append(name)

    root = '/home/liwang/data/face_raw_data'
    # all_img_list = []
    # all_img_list = get_filelist(root, all_img_list, ['.jpg', '.jpeg', '.webp', '.png', '.gif', '.svg'])

    img_list= glob('/home/liwang/data/face_raw_data/Goatee/*')
    print('img num: {}'.format(len(img_list)))
    img_list.extend(glob('/home/liwang/data/face_raw_data/Sideburns/*'))
    print('img num: {}'.format(len(img_list)))
    img_list.extend(glob('/home/liwang/data/face_raw_data/curly_beard/*'))
    print('img num: {}'.format(len(img_list)))
    img_list.extend(glob('/home/liwang/data/face_raw_data/山羊胡/*'))
    print('img num: {}'.format(len(img_list)))
    img_list.extend(glob('/home/liwang/data/face_raw_data/络腮胡/*'))
    print('img num: {}'.format(len(img_list)))
    img_list.extend(glob('/home/liwang/data/face_raw_data/连鬓胡/*'))
    print('img num: {}'.format(len(img_list)))
    all_img_list = img_list
    for img_path in all_img_list:
        
        img_name = get_img_name(img_path)

        # img_name = os.path.basename(img_path)

        input_img_list.append(img_path)
        input_img_name_list.append(img_name)
    
    # from ipdb import set_trace; set_trace()
    
    lmdb_cerater = LMDBCreater(input_img_list, input_img_name_list, lmdb_save_path)
    lmdb_cerater.run()