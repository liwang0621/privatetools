'''
Author       : Li Wang
Date         : 2022-04-12 10:38:44
LastEditors  : Li Wang
LastEditTime : 2022-04-14 23:53:09
FilePath     : /privatetools/data_processing/annos_processing/human_part_processing.py
Description  : 
'''
import os
from tqdm import tqdm
from loguru import logger

from utils import load_annos, save_annos


def get_dict(annos, label, name_list=None):
    out = {}
    if name_list is None:
        for anno in tqdm(annos):
            name = anno['img_name']
            for box in anno['bboxes']:
                box[-1] = label
            out[name] = anno
    else:
        for anno in tqdm(annos):
            name = anno['img_name']
            if name in name_list:
                for box in anno['bboxes']:
                    box[-1] = label
                out[name] = anno
    return out



if __name__=='__main__':
    annos_path_1 = '/home/liwang/projects/privatetools/data_processing/pseudo_annotation/demo_40000.txt'
    annos_path_2 = '/home/liwang/projects/privatetools/data_processing/pseudo_annotation/demo_30000.txt'

    annos = load_annos(annos_path_1)
    annos.extend(load_annos(annos_path_2))

    logger.info('human part annos num: {}'.format(len(annos)))

    name_list = []
    for anno in annos:
        name = anno['img_name']
        name_list.append(name)

    logger.info('load hand annos')
    hand_annos = load_annos('/home/liwang/data/human_part_det/yydata/anno_hand_7w.txt')
    logger.info('processing hand annos to dict')
    hand_annos = get_dict(hand_annos, label=4, name_list=name_list)

    logger.info('load face annos')
    face_annos = load_annos('/home/liwang/data/yy-data/lmdb/face_anno_raw/train_val/anno_train_7w.txt')
    logger.info('processing face annos to dict')
    face_annos = get_dict(face_annos, label=3, name_list=name_list)
    from IPython import embed; embed()

    logger.info('mearge annos')
    for anno in tqdm(annos):
        name = anno['img_name']
        face_bboxes = face_annos[name]['bboxes']
        hand_bboxes = hand_annos[name]['bboxes']

        anno['bboxes'].extend(face_bboxes)
        anno['bboxes'].extend(hand_bboxes)
    
    save_annos(annos=annos, save_anno_path='/home/liwang/data/human_part_det/yydata/human_part_yydata_det.txt')


    # # face_annos太大了
    # hand_annos = load_annos('/home/liwang/data/human_part_det/yydata/anno_hand_7w.txt')
    # name_list = []
    # for anno in hand_annos:
    #     name = anno['img_name']
    #     name_list.append(name)

    # logger.info('load face annos')
    # annos = load_annos('/home/liwang/data/yy-data/lmdb/face_anno_raw/anno.txt')
    # new_annos = []
    # for anno in tqdm(annos):
    #     name = anno['img_name']
    #     if name in name_list:
    #         new_annos.append(anno)
    
    # save_annos(annos=new_annos, save_anno_path='/home/liwang/data/yy-data/lmdb/face_anno_raw/train_val/anno_train_7w.txt')