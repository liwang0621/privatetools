import os
import cv2
import sys
import lmdb 
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm
from loguru import logger


sys.path.append('../')
from annos_processing.utils import load_annos, get_filelist


lmdb_save_path = '/home/liwang/data/product_det/dianshang_product/lmdb'
if not os.path.isdir(lmdb_save_path):
    os.makedirs(lmdb_save_path)


def write_lmdb(train_image_list):
    num = len(train_image_list)
    env = lmdb.open(lmdb_save_path,  map_size=int(1e12))
    cache = {}
    for idx, (img_name, img_path) in tqdm(enumerate(train_image_list)):
        imgKey = img_name.encode()
        img = cv2.imread(img_path)
        if img is None:
            logger.info('there is a problom!!!!!!!')
            continue

        cache[imgKey] = cv2.imencode(".jpg", img)[1].tobytes()

        if (idx + 1) % 10000 == 0:
           with env.begin(write=True) as txn:
               for k, v in cache.items():
                   txn.put(k, v)
           cache = {}
        elif idx == num - 1:
            with env.begin(write=True) as txn:
               for k, v in cache.items():
                   txn.put(k, v)

    env.close()


if __name__ == '__main__':
    root = '/home/liwang/data/product_data/检测训练数据/images/'

    logger.info('开始遍历全部数据！！！')

    annos1 = load_annos('/home/liwang/data/product_det/dianshang_product/test.txt')
    annos2 = load_annos('/home/liwang/data/product_det/dianshang_product/val.txt')
    annos3 = load_annos('/home/liwang/data/product_det/dianshang_product/train.txt')[:500000]
    annos = annos1 + annos2 + annos3

    train_image_list = []
    for anno in annos:
        img_name = anno['img_name']
        img_path = os.path.join(root, img_name)
        if os.path.exists(img_path):
            train_image_list.append([img_name, img_path])



    # from ipdb import set_trace; set_trace()


    # Filelist = []
    # get_filelist(root, Filelist, search_exts=['.jpeg', '.jpg'])
    # logger.info(f'遍历全部数据完成, Files num: {len(Filelist)}')
    # train_image_list = []
    # for img_path in Filelist:
    #     img_name = img_path.split(root)[-1]
    #     train_image_list.append([img_name, img_path])


    logger.info('开始写入lmdb: {}'.format(lmdb_save_path))
    cores = multiprocessing.cpu_count()
    logger.info('cores num: {}'.format(cores))
    pool = Pool(cores)
    pool.apply_async(write_lmdb, [train_image_list])
    pool.close()
    pool.join()