import os
import cv2
import json
import lmdb
import numpy as np
from random import random

from tqdm import tqdm

import sys

sys.path.append('../')
from annos_processing.utils import save_annos


def draw_box(img, boxes):
    b = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    r = np.random.randint(0, 256)
    color = (b, g, r)
    for box in boxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img


def random_generate_box(img_w, img_h):
    xs = [random() * img_w, random() * img_w]
    ys = [random() * img_h, random() * img_h]
    x1 = int(min(xs))
    y1 = int(min(ys))
    x2 = int(max(xs))
    y2 = int(max(ys))
    return [x1, y1, x2, y2]


def get_max_iou(box, dets):
    box = np.array(box)
    dets = np.array(dets)
    assert box.ndim == 1 and box.shape[0] == 4
    assert dets.ndim == 2 and dets.shape[1] == 4

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)

    x11 = np.maximum(box[0], x1)  # calculate the points of overlap
    y11 = np.maximum(box[1], y1)
    x22 = np.minimum(box[2], x2)
    y22 = np.minimum(box[3], y2)

    w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
    h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

    overlaps = w * h
    ious = overlaps / \
        (((box[2] - box[0]+1)*(box[3]-box[1]+1))+areas - overlaps)
    return ious.max()


def expand_bbox(box, image_w, image_h, expand_ratio):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    expend_size = max(w, h) * (1 + expand_ratio)

    x1_ex = max(0, cx - expend_size / 2)
    y1_ex = max(0, cy - expend_size / 2)
    x2_ex = min(cx + expend_size / 2, image_w)
    y2_ex = min(cy + expend_size / 2, image_h)

    return (int(x1_ex), int(y1_ex), int(x2_ex), int(y2_ex))


def get_affine_transform(box, out_shape):
    x, y, x2, y2 = box
    bboxW = x2 - x
    bboxH = y2 - y
    max_boxwh = max(bboxH, bboxW)
    bboxW = max_boxwh
    bboxH = max_boxwh
    outW, outH = out_shape

    trans = [[outW / bboxW, 0, outW / 2 - (outW / bboxW) * (x + bboxW / 2)],
             [0, outH / bboxH, outH / 2 - (outH / bboxH) * (y + bboxH / 2)]]
    return np.array(trans)


def load_annotations(ann_file,
                     lmdb_file,
                     min_obj_scale=1 / 32,
                     is_clip=False,
                     enable_neg=True,
                     use_ignore=False,
                     neg_sample_ratio=0.3,
                     neg_iou_th=0.0,
                     neg_label=1,
                     support_labels=None):
    support_labels = [] if not support_labels else support_labels
    annos = []
    with open(ann_file, "r") as f:
        for line in tqdm(f.readlines()):
            info = line.strip().split()
            img_name = info[0]
            img_w = int(info[1])
            img_h = int(info[2])
            obj_num = int(info[3])
            bboxes = []
            labels = []
            bboxes_ignore = []
            labels_ignore = []
            for obj_idx in range(0, obj_num):
                x1 = float(info[4 + obj_idx * 5])
                y1 = float(info[5 + obj_idx * 5])
                x2 = float(info[6 + obj_idx * 5])
                y2 = float(info[7 + obj_idx * 5])
                label = int(info[8 + obj_idx * 5])
                if x2 <= x1 or y2 <= y1:
                    continue
                if is_clip:  # clip bbox outside the image
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_w, x2)
                    y2 = min(img_h, y2)
                bbox = [x1, y1, x2, y2]
                obj_scale = max(x2 - x1, y2 - y1) / float(max(img_w, img_h))
                if label > 0 and obj_scale >= min_obj_scale:
                    labels.append(label - 1)
                    bboxes.append(bbox)
                else:
                    labels_ignore.append(label - 1)
                    bboxes_ignore.append(bbox)

            # 产生负样本
            if enable_neg:
                bboxes_all = []
                if use_ignore:
                    bboxes_all.extend(bboxes)
                    bboxes_all.extend(bboxes_ignore)
                else:
                    bboxes_all.extend(bboxes)
                expand_bboxes_all = []
                for box in bboxes_all:
                    box = expand_bbox(box, img_w, img_h, 0.2)
                    expand_bboxes_all.append(box)

                pos_sample_num = len(expand_bboxes_all)
                if pos_sample_num == 0:
                    neg_sample_num = 2
                else:
                    neg_sample_num = int(
                        (pos_sample_num /
                         (1 - neg_sample_ratio)) * neg_sample_ratio)

                skip_num = 100
                neg_bboxes = []
                while neg_sample_num and skip_num:
                    skip_num -= 1
                    gen_box = random_generate_box(img_w, img_h)
                    expand_gen_box = expand_bbox(gen_box, img_w, img_h, 0.2)
                    if pos_sample_num > 0:
                        if get_max_iou(expand_gen_box,
                                       expand_bboxes_all) <= neg_iou_th:
                            neg_bboxes.append(gen_box)
                            neg_sample_num -= 1
                    else:
                        neg_bboxes.append(gen_box)
                        neg_sample_num -= 1

            for box, label in zip(bboxes, labels):
                bboxes_pf = np.asarray([box], dtype=np.float32).reshape(
                    (-1, 4))
                labels_pf = np.asarray([label])
                annos.append(
                    dict(filename=img_name,
                         width=-1,
                         height=-1,
                         lmdb_file=lmdb_file,
                         ann=dict(bboxes=bboxes_pf,
                                  labels=labels_pf,
                                  bboxes_ignore=np.asarray(
                                      [], dtype=np.float32).reshape((-1, 4)),
                                  labels_ignore=np.asarray([])),
                         support_labels=support_labels))
            if use_ignore:
                for box, label in zip(bboxes_ignore, labels_ignore):
                    bboxes_pf = np.asarray([box], dtype=np.float32).reshape(
                        (-1, 4))
                    labels_pf = np.asarray([0])  # 默认只有一个类别
                    annos.append(
                        dict(filename=img_name,
                             width=-1,
                             height=-1,
                             lmdb_file=lmdb_file,
                             ann=dict(bboxes=bboxes_pf,
                                      labels=labels_pf,
                                      bboxes_ignore=np.asarray(
                                          [], dtype=np.float32).reshape(
                                              (-1, 4)),
                                      labels_ignore=np.asarray([])),
                             support_labels=support_labels))
            if enable_neg:
                for box in neg_bboxes:
                    bboxes_pf = np.asarray([box], dtype=np.float32).reshape(
                        (-1, 4))
                    labels_pf = np.asarray([neg_label])
                    annos.append(
                        dict(filename=img_name,
                             width=-1,
                             height=-1,
                             lmdb_file=lmdb_file,
                             ann=dict(bboxes=bboxes_pf,
                                      labels=labels_pf,
                                      bboxes_ignore=np.asarray(
                                          [], dtype=np.float32).reshape(
                                              (-1, 4)),
                                      labels_ignore=np.asarray([])),
                             support_labels=support_labels))
    for anno in annos[:10]:
        print(anno)
    return annos


def expand_crop_bbox(box, image_w, image_h, expand_ratio, shift_ratio=0.5):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    expend_size = max(w, h) * (1 + expand_ratio)

    x1_ex = max(0, cx - expend_size / 2)
    y1_ex = max(0, cy - expend_size / 2)
    x2_ex = min(cx + expend_size / 2, image_w)
    y2_ex = min(cy + expend_size / 2, image_h)

    if random() < shift_ratio:
        shift_x = random() * (x1 - x1_ex)
        shift_y = random() * (y1 - y1_ex)
        x1_ex = x1_ex + shift_x
        y1_ex = y1_ex + shift_y
        x2_ex = x2_ex + shift_x
        y2_ex = y2_ex + shift_y

    return (int(x1_ex), int(y1_ex), int(x2_ex), int(y2_ex))


def save_lmdb_annos(annos,
                    lmdb_path,
                    txn_read,
                    out_dir,
                    anno_save_path,
                    shift_ratio=0.5,
                    out_shape=(256, 256),
                    expand_ratio_range=(0.2, 1)):
    expand_ratio_scale = expand_ratio_range[1] - expand_ratio_range[0]

    env = lmdb.open(lmdb_path, map_size=int(1e12))
    txn = env.begin(write=True)

    new_annos = []
    par = tqdm(total=len(annos))
    for anno_id, anno in enumerate(annos):
        # 获取原图
        img_name = anno['filename']
        img = txn_read.get(img_name.encode())
        img = bytearray(img)
        img = cv2.imdecode(np.asarray(img, dtype=np.uint8), cv2.IMREAD_COLOR)
        image_h, image_w = img.shape[:2]
        #获取原先anno
        box = anno['ann']['bboxes'][0]
        label = anno['ann']['labels'][0]
        # 抠图
        expand_ratio = random() * expand_ratio_scale + expand_ratio_range[0]
        crop_box = expand_crop_bbox(box, image_w, image_h, expand_ratio,
                                    shift_ratio)
        trans = get_affine_transform(crop_box, out_shape)
        point1 = [box[0], box[1], 1]
        point2 = [box[2], box[3], 1]
        new_point1 = trans @ np.array(point1)
        new_point2 = trans @ np.array(point2)
        new_box = [new_point1[0], new_point1[1], new_point2[0], new_point2[1]]
        crop_img = cv2.warpAffine(img,
                                  trans,
                                  out_shape,
                                  flags=cv2.INTER_LINEAR)
        # 写入lmdb
        save_name = os.path.splitext(img_name)[0] + f'_{anno_id}_{label}.jpg'
        img_data = cv2.imencode(".jpg", crop_img)[1].tobytes()
        txn.put(save_name.encode(), img_data)
        # 保存图片
        save_path = os.path.join(out_dir, save_name)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_crop_img = draw_box(crop_img.copy(), [new_box])
        cv2.imwrite(save_path, save_crop_img)

        # 新的anno
        if label == 0:
            new_box.append(1)
            new_anno = {
                'img_name': save_name,
                'bboxes': [new_box],
                'img_w': image_w,
                'img_h': image_h
            }
        else:
            new_anno = {
                'img_name': save_name,
                'bboxes': [],
                'img_w': image_w,
                'img_h': image_h
            }
        new_annos.append(new_anno)
        par.update(1)
    txn.commit()
    env.close()

    assert len(annos) == len(new_annos)
    save_annos(new_annos, anno_save_path)


def main():
    ann_file = '/home/liwang/data/face_det/widerface/val/anno.txt'
    lmdb_file = '/home/liwang/data/face_det/widerface/val/lmdb'
    annos = load_annotations(ann_file, lmdb_file)

    out_dir = '/home/liwang/data/pf_det/widerface/val/images'
    anno_save_path = '/home/liwang/data/pf_det/widerface/val/anno.txt'
    lmdb_path = '/home/liwang/data/pf_det/widerface/val/lmdb'
    env_db = lmdb.open(lmdb_file, readonly=True, lock=False)
    txn_read = env_db.begin()

    save_lmdb_annos(annos, lmdb_path, txn_read, out_dir, anno_save_path)

    # from IPython import embed
    # embed()


if __name__ == '__main__':
    main()