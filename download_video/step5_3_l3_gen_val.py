#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
This module 

Authors: DaiBing(daibing@baidu.com)
Date: 2019-06-17 14:46:56
"""
import base64
import os
import pwd
import sys
import multiprocessing
import json
import shutil
import random
import argparse
import numpy as np
import pandas as pd

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
    with open(file_name, "r", encoding='utf-8') as f:
    # with open(file_name, "r") as f:
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
            if len(split_str):
                line = split_str.join(file_list[i])
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
    # with open(index_txt, "r") as f:
    with open(index_txt, "r") as f:
        lines = f.readlines()
    id_label_map = {}
    label_id_map = {}
    for i in range(len(lines)):
        items = lines[i].strip().split("\t")
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
        
def lst_to_map(val_lst, idx=0):
    val_map = {}
    for item in val_lst:
        if val_map.get(item[idx], None) == None:
            val_map[item[idx]] = [item]
        else:
            val_map[item[idx]].append(item)
    return val_map
def insert_map(src_map, key, value):
    if src_map.get(key, None) == None:
        src_map[key] = [value]
    else:
        src_map[key].append(value)
    return
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
lst_to_map(val_lst, idx=0)
insert_map(src_map, key, value)
for i, item in enumerate(src_lst):
"""

# parser = argparse.ArgumentParser(description='entity fc score combine')
# parser.add_argument('--dst_root', default="",type=str, help='')
# args = parser.parse_args()
def print_index_columns(data, sign=""):
    indexs = data._stat_axis.values.tolist()  # 行名称
    columns = data.columns.values.tolist()  # 列名
    print("文件名", sign, "行索引数", len(indexs), "前2行",indexs[:2],"最后2行", indexs[-2:])
    print("文件名",sign, "列索引数", len(columns), "前2列", columns[:2], "最后2列", columns[-2:])
    return indexs, columns

def write_file(file_list, save_file, split_str="\t"):
    """
    删掉原来的函数
    """
    with open(save_file, "w", encoding='utf-8') as f:
        for i in range(len(file_list)):
            file_list[i] = [str(l) for l in file_list[i]]
            if len(split_str):
                line = split_str.join(file_list[i])
            else:
                line = file_list[i]
            f.write(line + "\n")
    return

def read_csv(csv_file=""):
    val_lst = read_txt(csv_file)
    columns = val_lst[0][1:]
    indexs = [l[0] for l in val_lst[1:]]
    data = val_lst[1:]
    data = [l[1:] for l in data]
    df = pd.DataFrame(data, columns=columns, index=indexs)
    print_index_columns(df, sign="")
    return df

def write_df(df,csv_file="tmp.csv"):
    indexs, colums = print_index_columns(df)
    data = df.values.tolist()
    res_lst = []
    res_lst.append(["ID"] + colums)
    assert(len(data) == len(indexs))
    for i in range(len(data)):
        res_lst.append([indexs[i]] + data[i])
    write_file(res_lst, csv_file)


def func(item):
    count = 0
    for l in item[1:]:
        if int(l) == 0:
            count += 1
    if count < (len(item) - 1) * 0.5:
        return True
    else:
        return False
import random
def v1_to_v2(final_lab):
    if final_lab == "Skincare & Makeup Product&Display":
            final_lab = "Skincare & Makeup Product & Display"
    if final_lab == "Mukbangs & Tasting":
        final_lab = "Mukbangs & Review"
    if final_lab == "Diary & VLOG":
        final_lab = "Diary & Vlog"
    if final_lab == "Other Talent":
        final_lab = "Other Art & Talent"
    if final_lab == "Finger Dance & Basic Dance":
        final_lab = "Finger Dance"
    if final_lab == "Anime & Comics":
        final_lab = "Comics"
    if final_lab == "hair":
        final_lab = "Hair"
    if final_lab == "Other Fashion":
        final_lab = "Other fashion"
    if final_lab == "Special Makeup & Dress":
        final_lab = "Special Makeup & Dress-up"
    if final_lab == "Outfits":
        final_lab = "Dressing Outfits"
    if final_lab == "Lip":
        final_lab = "Lip-sync"
    if final_lab == "Scripted Comedy":
        final_lab = "Scripted Drama"
    if final_lab == "Other Comedy":
        final_lab = "Hilarious Stories"
    if final_lab == "Pranks":
        final_lab = "Hilarious Stories"
    if final_lab == "Hilarious Fails":
        final_lab = "Hilarious Stories"
    if final_lab == "Celebrity Clips":
        final_lab = "Celebrity & Star"
    return final_lab
import traceback
from mmcv.fileio import FileClient
import io
import decord
error_map={}
def func(file_lst, feat_path, videos_path, total_txt, j):
    io_backend='disk'
    file_client = FileClient(io_backend)
    print("file_lst", len(file_lst))
    good_txt = total_txt + "__" + str(j)
    good_lst = []
    for i, item in enumerate(file_lst):
        try:
            mp4 = item[0]
            fea = mp4[:-4] + ".npy"
            feat_file = os.path.join(feat_path, fea)
            video_file = os.path.join(videos_path, mp4)
            audio = np.load(feat_file)
            audio_shape = audio.shape
            if os.path.exists(video_file):
                file_obj = io.BytesIO(file_client.get(video_file))
                container = decord.VideoReader(file_obj, num_threads=8)
                c_type = type(container)
                good_lst.append(item)
            if i % 10000 == 0:
                print("process", j, "i", i, len(file_lst))
        except Exception as e:
            traceback.print_exc()
            print("ERROR", item)
            error_map[mp4] = 1
    print(good_txt)
    print(len(good_lst))
    write_file(good_lst, good_txt, split_str=" ")
import argparse
if __name__ == '__main__':
    
    #L333w
    #根据抽取的音频，生成val，第一列vid_mp4,第二列labid，空格隔开
    # feature_path = "/home/daib/data/video_cls/l1_l2_tags/L3_lab34_33w/audio_features"
    # url_path = "/home/daib/data/video_cls/l1_l2_tags/L3_lab34_33w/L3_330085_vid_url_l1_l2_l3_cap.csv"#\t隔开，第一列是vid，第二列是url
    # #标签映射配置
    # lab_txt = "/home/daib/data/video_cls/l1_l2_tags/L3_lab34_33w/L3_labid_lab_num34.txt"
    # id_label_map, label_id_map = txt_to_map(lab_txt)
    # print("id_label_map", id_label_map)
    # print("label_id_map", label_id_map)
    # assert(len(label_id_map.keys()) == len(id_label_map.keys()))

    # #输出
    # val_txt = "/home/daib/data/video_cls/l1_l2_tags/L3_lab34_33w/l2_anno.txt.total"#根据audio抽帧成功的，生成val格式，第一列vid_mp4,第二列labid，空格隔开
    # exist_txt = url_path + "_exists"#处理成功的视频
    # exist_lst = []

    #l2,通过特征文件，
    parser = argparse.ArgumentParser()
    parser.add_argument("--url_path", type=str, default="",help="path of url file")
    parser.add_argument("--root_path", type=str, default="",help="path of url file")
    parser.add_argument("--videos_path", type=str, default="",help="path of url file")
    parser.add_argument("--audios_path", type=str, default="",help="path of url file")
    parser.add_argument("--audio_features_path", type=str, default="",help="path of url file")
    parser.add_argument("--caption_path", type=str, default="",help="path of url file")
    parser.add_argument("--lab_txt", type=str, default="/home/daib/data/l1_l2_tags/87L2_label_index.txt",help="path of url file")
    parser.add_argument("--total_txt", type=str, default="",help="path of url file")

    parser.add_argument("--shorter_size", type=int, default=256)
    parser.add_argument("--process_num", type=int, default=48)
    parser.add_argument("--vid_idx", type=int, default=0)
    parser.add_argument("--url_idx", type=int, default=1)
    parser.add_argument("--l1_lab_idx", type=int, default=2)
    parser.add_argument("--l2_lab_idx", type=int, default=3)
    parser.add_argument("--cap_idx", type=int, default=4)
    parser.add_argument("--date", type=str, default="",help="path of url file")
    
    args = parser.parse_args()
    # args.url_path = "/home/daib/data/video_cls/l1_l2_tags/20220805_20w/vid_url_l1_l2_cap_20220802_0808_35w.csv"#\t隔开，第一列是vid，第二列是url
    # args.root_path = "/home/daib/data/video_cls/l1_l2_tags/20220805_20w "
    
    args.videos_path = os.path.join(args.root_path, "videos")
    args.audios_path = os.path.join(args.root_path, "audios")
    args.audio_features_path = os.path.join(args.root_path, "audio_features")
    args.caption_path = os.path.join(args.root_path, "caption.total.csv")
    
    # feature_path = "/home/daib/data/video_cls/l1_l2_tags/20220805_20w/audio_features"
    # url_path = "/home/daib/data/video_cls/l1_l2_tags/20220805_20w/20220805_vid_url_l1_l2_cap.csv"#\t隔开，第一列是vid，第二列是url
    # vid_idx = 0
    # url_idx = 1
    # l1_lab_idx = 2
    # lab_idx = 3
    # cap_idx = 4
    #标签映射配置
    # lab_txt = "/home/daib/data/video_cls/l1_l2_tags/87L2_label_index.txt"
    id_label_map, label_id_map = txt_to_map(args.lab_txt)
    print("id_label_map", id_label_map)
    print("label_id_map", label_id_map)
    assert(len(label_id_map.keys()) == len(id_label_map.keys()))

    #输出
    exist_lst = []
    vid_lst = read_txt(args.url_path)
    val_lst = []
    for item in vid_lst:
        vid = item[args.vid_idx]
        mp4 = vid + ".mp4"
        manul_lab = item[args.l2_lab_idx]#如果没有人工标注，这里需要指定
        manul_lab = v1_to_v2(manul_lab)
        if manul_lab == "Non" and item[args.l1_lab_idx] == "Games":
            manul_lab = "Non-Video games"
        if label_id_map.get(manul_lab, None) is None:
            print("Lab ERROR", item)
            continue
        manul_labid = label_id_map[manul_lab]
        feature_name = mp4.replace(".mp4", ".npy")
        feature_file = os.path.join(args.audio_features_path, feature_name)#用抽特征测试是否存在
        if os.path.exists(feature_file):
            val_lst.append([mp4, manul_labid])
            exist_lst.append(item)
        else:
            pass
            # print("ERROR no feature_file!!!",item)
    num_w = len(val_lst) / 10000#以w作为单位
    total_name = "l2_anno.txt.total_{}_{}w.txt".format(args.date, num_w)#"l2_anno.txt.total" + args.url_path.split("vid_url_l1_l2_cap")[-1].replace(".csv",".txt")
    args.total_txt = os.path.join(args.root_path, total_name)

    print("args.total_txt", args.total_txt, len(val_lst) )
    write_file(val_lst, args.total_txt, split_str=" ")
    exist_txt = args.url_path + "_{}w".format(num_w)#处理成功的视频
    print("exist_txt", exist_txt, len(exist_lst))
    write_file(exist_lst, exist_txt, split_str="\t")

    #多进程检查训练样本是否合法
    val_txt = args.total_txt
    feat_path = os.path.join(args.root_path, "audio_features")
    video_path = os.path.join(args.root_path, "videos")
    val_lst = read_txt(val_txt, split_str=" ")
    print("val_lst", len(val_lst))
    file_lst = val_lst
    num_process = args.num_process
    fileArr = [[]for i in range(num_process)]
    for i in range(len(file_lst)):
        fileArr[i % num_process].append(file_lst[i])
    pool = multiprocessing.Pool()
    for j in range(num_process):
        pool.apply_async(func, args=(fileArr[j], feat_path, video_path, val_txt, j))
    pool.close()
    pool.join()
    bak_txt = val_txt.replace("/l2_anno.txt.", "/store/l2_anno.txt.")
    mkdir_path(bak_txt)
    cmd = "cat {}__* > {}.good && mv {} {} && mv {}.good {} && rm -rf {}__* && cat {}|wc -l"\
        .format(val_txt, val_txt, val_txt, bak_txt, val_txt, val_txt, val_txt, val_txt)
    print(cmd)
    os.system(cmd)
    print("src val_lst", bak_txt, len(val_lst))
    good_lst = read_txt(val_txt, split_str=" ")
    print("good val_txt", val_txt, len(good_lst))

