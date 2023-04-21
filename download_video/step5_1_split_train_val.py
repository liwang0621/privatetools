#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
This module 

Authors: DaiBing(daibing@baidu.com)
Date: 2022-01-19 11:04:52
"""
import base64
import os
import sys
import multiprocessing
import json
import shutil
import random
import argparse
import numpy as np
import traceback

parser = argparse.ArgumentParser(description='entity fc score combine')
parser.add_argument('--dst_root', default="/home/daibing/dataset/sport_cls72_addEntity_preresize1.44_resize256",
                    type=str, help='save_')

def mkdir_path(dst_file):
    """
    新建目标文件的路径
    :param dst_file:目标文件
    :return:
    """
    dst_path, dst_name = os.path.split(dst_file)
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    return dst_path, dst_name

def copy_file(src_file, dst_file):
    """
    拷贝文件
    :param src_file: 源文件
    :param dst_file: 目标文件
    :return:
    """
    try:
        mkdir_path(dst_file)
        if os.path.exists(dst_file):  # 目标文件存在
            print("ERROR:dst_file exists!",dst_file)
            return
        if not os.path.exists(src_file):  # 源文件不存在
            print("ERROR:src_file not exists!", src_file)
            return
        shutil.copy(src_file, dst_file)
    except Exception as e:
        print("ERROR:", str(e), src_file, dst_file)
    return
def print_index_columns(data, sign=""):
    indexs = data._stat_axis.values.tolist()  # 行名称
    columns = data.columns.values.tolist()  # 列名
    print(sign, "行索引", len(indexs), indexs[:2], indexs[-2:])
    print(sign, "列索引", len(columns), columns[:2], columns[-2:])
    return indexs, columns

def move_file(src_file, dst_file):
    """
    移动文件
    :param src_file: 源文件
    :param dst_file: 目标文件
    :return:
    """
    try:
        mkdir_path(dst_file)
        if os.path.exists(dst_file):  # 目标文件存在
            print("ERROR:dst_file exists!",dst_file)
            return
        if not os.path.exists(src_file):  # 源文件不存在
            print("ERROR:src_file not exists!", src_file)
            return
        shutil.move(src_file, dst_file)
    except Exception as e:
        print("ERROR:", str(e), src_file, dst_file)
    return

def get_path_list(root_path):
    """
    不递归获取目录下所有文件夹名称，
    :param root_path: 目录路径
    :return:
    """
    tmp_list = os.listdir(root_path)
    res_list = []
    for path in tmp_list:
        if os.path.isdir(os.path.join(root_path, path)):
            res_list.append(path)
    try:#如果是标签id，进行排序后输出
        a = int(res_list[0])
        res_list = [int(l) for l in res_list]
        res_list = sorted(res_list)
        res_list = [str(l) for l in res_list]
        return res_list
    except:
        pass
    return res_list

def get_file_list(root_path, suffix=[".jpg"]):
    """
    获取目录下所有满足后缀在suffix中的文件全路径
    :param path: 目录
    :param suffix: 后缀列表
    :return:
    """
    res = []
    for fpathe, dirs, fs in os.walk(root_path):
        for f in fs:
            for sf in suffix:
                if f.endswith(sf):
                    res.append(os.path.join(fpathe, f))
    return res

def get_name_list(path, suffix=[".jpg"]):
    """
    获取目录下所有满足后缀在suffix中的文件名
    :param path:
    :param suffix:
    :return:
    """
    res = []
    for fpathe, dirs, fs in os.walk(path):
        for f in fs:
            for sf in suffix:
                if f.endswith(sf):
                    res.append(f)
    return res

def read_txt(file_name, split_str="\t"):
    """
    读取txt文件
    :param file_name: 文件名
    :param split_str: 分割符
    :return:
    """
    with open(file_name, "r") as f:
        lines = f.readlines()
    items = []
    for line in lines:
        if len(split_str):
            item = line.strip().split(split_str)
        else:
            item = line.strip()
        items.append(item)
    return items

def write_file(file_list, save_file, split_str="\t"):
    """
    将列表写入文件
    :param file_list: 列表
    :param save_file: 保存文件
    :param split_str: 分割符
    :return:
    """
    with open(save_file, "w") as f:
        for i in range(len(file_list)):
            file_list[i] = [str(l) for l in file_list[i]]
            if len(split_str):
                try:
                    line = split_str.join(file_list[i])
                except:
                    traceback.print_exc()
            else:
                line = file_list[i]
            f.write(line + "\n")
    return

def write_label_jpg_map(label_jpg_map, file_name):
    """
    将字典写入文件
    :param label_jpg_map: key为label，value为图像路径列表
    :param file_name:
    :return:
    """
    with open(file_name, "w") as f:
        for key in label_jpg_map:
            jpg_list = label_jpg_map[key]
            for jpg_file in jpg_list:
                f.write("{} {}\n".format(jpg_file, key))
    return

def random_topk(res_list, topK=1):
    """
    随机从list中取前k个
    :param res_list: 列表
    :param topK: 超参
    :return:
    """
    random.shuffle(res_list)
    if topK > len(res_list):
        return res_list
    res_list = res_list[:topK]
    return res_list

def txt_to_map(index_txt):
    """
    输入标签文件路径，输出id_label_map, label_id_map
    :param index_txt: 标签文件路径
    :return:
    """
    with open(index_txt, "r") as f:
        lines = f.readlines()
    id_label_map = {}
    label_id_map = {}
    for i in range(len(lines)):
        items = lines[i].strip().split(" ")
        if len(items) == 0:
            continue
        lab_id = str(items[0])
        lab = str(items[1])
        id_label_map[lab_id] = lab
        label_id_map[lab] = lab_id
    return id_label_map, label_id_map


def copy_files(src_files, src_root, dst_root):
    """
    批量文件拷贝
    :param src_files: 源文件列表
    :param src_root: 源文件根目录
    :param dst_root: 目标根目录
    :return:
    """
    for src_file in src_files:
        dst_file = src_file.replace(src_root, dst_root)
        copy_file(src_file, dst_file)
"""
mkdir_path(dst_file)
txt_to_map(index_txt)
random_topk(res_list, topK=1)
write_label_jpg_map(label_jpg_map, file_name)
write_file(file_list, save_file, split_str="\t")
read_txt(file_name, split_str="\t")
get_name_list(path, suffix=[".jpg"])
get_file_list(root_path, suffix=[".jpg"])
get_path_list(root_path)
move_file(src_file, dst_file)
copy_file(src_file, dst_file)
copy_files(src_files, src_root, dst_root)
"""
def table_to_lst(table):
    src_lst = []
    for rowx in range(0, table.nrows):
        lst = []
        for colx in range(table.ncols):
            lst.append(table.cell(rowx, colx).value)
        src_lst.append(lst)
    # print(src_lst)
    return src_lst
def get_lst_from_xlsx(file, sheet_idx = 0):
    #
    data = xlrd.open_workbook(file)
    table = data.sheets()[sheet_idx]#第1个sheet
    lst = table_to_lst(table)
    return lst
def lst_to_map(val_lst, idx=0):
    val_map = {}
    for item in val_lst:
        if val_map.get(item[idx], None) == None:
            val_map[item[idx]] = [item]
        else:
            val_map[item[idx]].append(item)
    return val_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--url_path", type=str, default="",help="path of url file")
    parser.add_argument("--root_path", type=str, default="",help="path of url file")
    parser.add_argument("--videos_path", type=str, default="",help="path of url file")
    parser.add_argument("--audios_path", type=str, default="",help="path of url file")
    parser.add_argument("--audio_features_path", type=str, default="",help="path of url file")
    parser.add_argument("--caption_path", type=str, default="",help="path of url file")
    parser.add_argument("--lab_txt", type=str, default="/home/daib/data/video_cls/l1_l2_tags/87L2_label_index.txt",help="path of url file")
    parser.add_argument("--total_txt", type=str, default="",help="path of url file")

    parser.add_argument("--shorter_size", type=int, default=256)
    parser.add_argument("--process_num", type=int, default=48)
    parser.add_argument("--vid_idx", type=int, default=0)
    parser.add_argument("--url_idx", type=int, default=1)
    parser.add_argument("--l1_lab_idx", type=int, default=2)
    parser.add_argument("--l2_lab_idx", type=int, default=3)
    parser.add_argument("--cap_idx", type=int, default=4)
    parser.add_argument("--train_thd", type=float, default=0.95)
    parser.add_argument("--val_num", type=int, default=0)

    args = parser.parse_args()
    # args.url_path = "/home/daib/data/video_cls/l1_l2_tags/20220805_20w/vid_url_l1_l2_cap_20220802_0808_35w.csv"#\t隔开，第一列是vid，第二列是url
    # args.root_path = "/home/daib/data/video_cls/l1_l2_tags/20220805_20w "
    
    args.videos_path = os.path.join(args.root_path, "videos")
    args.audios_path = os.path.join(args.root_path, "audios")
    args.audio_features_path = os.path.join(args.root_path, "audio_features")
    args.caption_path = os.path.join(args.root_path, "caption.total.csv")
    args.total_txt = os.path.join(args.root_path, "l2_anno.txt.total")


    # 代码说明，将目录下得标签文件(带后缀_clinical.txt)和01pair的文件进行结合,在最后一列增加标签,生成得到args.label_txt，后缀为label01
    ###############参数区域#####################
    #args.label_txt = "data/p2_d3/GSE112840Expesccimmdel_media_pair01_label01_thd0.7.csv"  # 第三步生成的csv文件，注意后缀是csv，label01后缀表示加入标签
    #args.train_thd = 0.7  # 训练集的比例，默认为0.7
    ###############参数区域#####################
    ###############参数区域#####################
    #args.label_txt = "data/p2_d3/GSE122497Expesccimmdel_2_pair01_label01_thd0.8.csv"  # 第三步生成的csv文件，注意后缀是csv，label01后缀表示加入标签
    #args.train_thd = 0.7  # 训练集的比例，默认为0.7
    ###############参数区域#####################
    ###############参数区域#####################
    # args.total_txt = "/home/daib/data/video_cls/l1_l2_tags/20220805_20w/l2_anno.txt.total"  # 第三步生成的csv文件，注意后缀是csv，label01后缀表示加入标签
    # args.train_thd = 0.95  # 训练集的比例，默认为0.7
    ###############参数区域#####################
    #第一步指定标注文件(带_pair01_filter0.8_label01.csv)后缀
    
    src_txt = args.total_txt
    src_lst = read_txt(src_txt)
    if args.val_num > 0:
        args.train_thd = (len(src_lst) - args.val_num)/ float(len(src_lst))
    # src_lst = src_lst[:4000]
    #这里只保留4000个样本
    print("src_txt", src_txt, len(src_lst), len(src_lst[0]),len(src_lst[1]),src_lst[0][0],src_lst[0][-1])
    rand_lst = src_lst#进行shuffle
    random.shuffle(rand_lst)
    # rand_lst = rand_lst[:3000]
    #生成0.7的训练集
    train_len = int(len(rand_lst) * args.train_thd)#训练集0.7，测试集0.3
    train_lst = rand_lst[:train_len]
    # train_lst = [l[1:] for l in train_lst]#去掉ID列名
    train_txt = src_txt.replace("total", "train")
    print("训练集路径", train_txt, "行数", len(train_lst), "列数", len(train_lst[0]))
    write_file(train_lst, train_txt, split_str=" ")
    #生成0.3的测试集
    test_lst = rand_lst[train_len:]
    # test_lst = [l[1:] for l in test_lst]#去掉ID列名
    test_txt = src_txt.replace("total", "val")
    print("测试集路径",  test_txt, "行数", len(test_lst),  "列数", len(test_lst[0]))#去掉ID列名
    write_file(test_lst, test_txt, split_str="\t")
    print()

