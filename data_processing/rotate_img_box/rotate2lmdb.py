'''
Author       : Li Wang
Date         : 2022-01-11 10:40:14
LastEditors  : Li Wang
LastEditTime : 2022-04-18 15:45:02
FilePath     : /privatetools/data_processing/rotate_img_box/rotate2lmdb.py
Description  : 
'''
import os
import cv2
import lmdb
import math
import random
import numpy as np
from tqdm import tqdm

def annostr2dict(annostr):
    info = annostr.strip().split()
    img_name = info[0]
    img_w = int(info[1])
    img_h = int(info[2])
    obj_num = int(info[3])
    bboxes = []
    labels = []
    
    for obj_idx in range(0, obj_num):
        x1 = float(info[4 + obj_idx * 5])
        y1 = float(info[5 + obj_idx * 5])
        x2 = float(info[6 + obj_idx * 5])
        y2 = float(info[7 + obj_idx * 5])
        label = int(info[8 + obj_idx * 5])
        if x2 <= x1 or y2 <= y1:
            continue
        bbox = [x1, y1, x2, y2]
        
        labels.append(label - 1)
        bboxes.append(bbox)
    out = {
        'img_name': img_name,
        'bboxes': bboxes
    }
    return out

def render(img, bboxes):
    b = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    r = np.random.randint(0, 256)
    color = (b, g, r)
    color = (255, 0 ,225)
    for box in bboxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        class_name = 'Face'
        y_text = y1 + 20
        render_text = "{}".format(class_name)
        cv2.putText(img, render_text, (int(x1), int(y_text)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    return img


def rotate_bound(image, angle):
    """

    :param image: 原图像
    :param angle: 旋转角度
    :return: 旋转后的图像
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    img = cv2.warpAffine(image, M, (nW, nH))
    # perform the actual rotation and return the image
    return img


def rotate_img_bbox(img, bboxes, angle=5, scale=1.):
    '''
    参考:https://blog.csdn.net/u014540717/article/details/53301195crop_rate
    输入:
        img:图像array,(h,w,c)
        bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        angle:旋转角度
        scale:默认1
    输出:
        rot_img:旋转后的图像array
        rot_bboxes:旋转后的boundingbox坐标list
    '''
    #---------------------- 旋转图像 ----------------------
    w = img.shape[1]
    h = img.shape[0]
    # 角度变弧度
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    # 仿射变换
    rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

    #---------------------- 矫正bbox坐标 ----------------------
    # rot_mat是最终的旋转矩阵
    # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
    rot_bboxes = list()
    for bbox in bboxes:
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        point1 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymin, 1]))
        point2 = np.dot(rot_mat, np.array([xmax, (ymin+ymax)/2, 1]))
        point3 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymax, 1]))
        point4 = np.dot(rot_mat, np.array([xmin, (ymin+ymax)/2, 1]))
        # 合并np.array
        concat = np.vstack((point1, point2, point3, point4))
        # 改变array类型
        concat = concat.astype(np.int32)
        # 得到旋转后的坐标
        rx, ry, rw, rh = cv2.boundingRect(concat)
        rx_min = rx
        ry_min = ry
        rx_max = rx+rw
        ry_max = ry+rh
        # 加入list中
        rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])

    return rot_img, rot_bboxes


def main():
    ann_file = '/home/liwang/data/yy-data/lmdb/face_anno/train_val/anno_val_labeled.txt'
    annos = []
    with open(ann_file, 'r') as f:
        for line in f.readlines():
            annos.append(line.strip())
            
    lmdb_path = '/home/liwang/data/yy-data/lmdb/lmdb'
    env_db_read = lmdb.open(lmdb_path, readonly=True, lock=False)
    txn_read = env_db_read.begin()

    rewrite_lmdb = True

    # write data
    angle = 90
    label = 1
    save_root = '/home/liwang/data/yy-data/lmdb/face_anno/train_val/val_rotate_{}'.format(angle)
    save_anno_path = os.path.join(save_root, "anno_label.txt")
    if rewrite_lmdb:
        save_lmdb_dir = os.path.join(save_root, "lmdb")
        if not os.path.isdir(save_lmdb_dir):
            os.makedirs(save_lmdb_dir)

        env = lmdb.open(save_lmdb_dir, map_size=int(1e12))
        txn = env.begin(write=True)
    with open(save_anno_path, "w") as f:
        for anno in tqdm(annos):
            anno = annostr2dict(anno)
            img_name = anno['img_name']
            bboxes = anno['bboxes']
            data = txn_read.get(img_name.encode())
            data = bytearray(data)
            img = cv2.imdecode(np.asarray(data, dtype=np.uint8),
                            cv2.IMREAD_COLOR)

            img, bboxes = rotate_img_bbox(img, bboxes, angle=angle)
            tmp = os.path.splitext(img_name)[-1]
            rotate_img_name = img_name.replace(tmp, '_{}{}'.format(angle, tmp))
            if rewrite_lmdb:
                img_data = cv2.imencode(".jpg", img)[1].tobytes()
                txn.put(rotate_img_name.encode(), img_data)
            img_h = img.shape[0]
            img_w = img.shape[1]

            img_anno_str = ' '
            for box in bboxes:
                assert len(box) == 4
                img_anno_str += '{} {} {} {} {} '.format(box[0], box[1], box[2], box[3], label)
                # print(img_anno_str)

            img_anno = rotate_img_name + " {} {} {}".format(img_w, img_h, len(bboxes)) + img_anno_str

            f.writelines(img_anno + '\n')

    env_db_read.close()
    if rewrite_lmdb:
        txn.commit()
        env.close()



if __name__=='__main__':
    main()