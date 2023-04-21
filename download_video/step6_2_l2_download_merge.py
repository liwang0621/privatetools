#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
This module 

Authors: DaiBing(daibing@baidu.com)
Date: 2019-06-17 14:46:56
"""
from asyncore import read
import base64
from distutils.spawn import spawn
import os
from re import sub
import sys
import multiprocessing
import json
import shutil
import random
import argparse
from xml.etree.ElementPath import ops
import numpy as np
import pandas as pd
import csv

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
            # print("ERROR:dst_file exists!",dst_file)
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
            # print("ERROR:dst_file exists!",dst_file)
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
        # if len(item) < 5:
        #     print("item", item)
        #     continue
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

parser = argparse.ArgumentParser(description='entity fc score combine')
parser.add_argument('--dst_root', default="",type=str, help='')
args = parser.parse_args()
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
    with open(save_file, "w") as f:
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


# def func(item):
#     count = 0
#     for l in item[1:]:
#         if int(l) == 0:
#             count += 1
#     if count < (len(item) - 1) * 0.5:
#         return True
#     else:
#         return False
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
    if final_lab == "anime & comics":
        final_lab = "Comics"
    if final_lab == "hair":
        final_lab = "Hair"
    if final_lab == "Other Relation":
        final_lab = "Other Relations"
    if final_lab == "anime":
        final_lab = "Comics"
    return final_lab

def write_caption_csv(cap_csv=None, line_dicts=None, keys=["video_id", "merged_caption", "play_url"]):
    with open(cap_csv, "w") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=keys)
        writer.writeheader()
        for row in line_dicts:
            video_id = row.get("video_id", None)
            caption = row.get("merged_caption", None)
            play_url = row.get("merged_caption", "")
            writer.writerow(dict(video_id=video_id, merged_caption=caption, play_url=play_url))


def read_caption_csv(cap_csv, keys=[]):
    """通过key读取dict
    """
    res_lst = []
    print("read_caption_csv", keys)
    with open(cap_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            line_dict = dict()
            for key in keys:
                line_dict[key] = row[key]
            res_lst.append(line_dict)
    return res_lst

def read_txt_to_map(txt, split_str="\t", idx=0):
    lst = read_txt(txt, split_str=split_str)
    lst_map = lst_to_map(lst, idx=idx)
    return lst_map
import traceback
error_map = {}
from mmcv.fileio import FileClient
import io
import decord
error_map={}

# def func(file_lst, root_path):
#     print("file_lst", len(file_lst))
#     feat_path = os.path.join(root_path, "audio_features")
#     video_path = os.path.join(video_path, "videos")
#     for i, item in enumerate(file_lst):
#         try:
#             mp4 = item[0]
#             fea = mp4[:-4] + ".npy"
#             fea_file = os.path.join(feat_path, fea)
#             audio = np.load(fea_file)
#             audio_shape = audio.shape
            
#             if i % 10000 == 0:
#                 print("i", i)
#         except Exception as e:
#             traceback.print_exc()
#             print("ERROR", item)
#             error_map[mp4] = 1
import subprocess
def ffmpeg_mp4_r5(src_file, dst_file, delete_src_file=False):
    try:
        mkdir_path(dst_file)
        if os.path.exists(dst_file):
            # print("WARNING!!! os.path.exists(dst_file)", dst_file)
            if os.path.exists(dst_file) and os.path.exists(src_file) and delete_src_file:
                os.remove(src_file)#删掉源文件
            return  
        cmd = "ffmpeg -loglevel fatal -nostdin -y -i {} -r 5 -vf scale=256:-2 -f mp4 -an {}".format(src_file, dst_file)
        r = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if os.path.exists(dst_file) and os.path.exists(src_file) and delete_src_file:
            os.remove(src_file)#删掉源文件
    except Exception as e:
        traceback.print_exc()

def ffmpeg_mp4_lst(videos_lst, videos_path, dst_path, j, delete_src_file):
    print("videos_lst", len(videos_lst))
    for i, src_file in enumerate(videos_lst):
        if i % 1000 == 0:
            print("process", j, "count", i, len(videos_lst))
        dst_file = src_file.replace(videos_path, dst_path)
        ffmpeg_resize_refps(src_file, dst_file, shorter_size=256, fps=5, delete_src_file=True)
        # ffmpeg_mp4_r5(src_file, dst_file, delete_src_file=delete_src_file)
    return
import ffmpeg

def ffmpeg_resize_refps(ori_video_path, save_video_path, shorter_size=256, fps=5, delete_src_file=True):
    if shorter_size <= 0:#不进行resize
        os.rename(ori_video_path, save_video_path)
    else:
        try:
            mkdir_path(save_video_path)
            if os.path.exists(ori_video_path) and os.path.exists(save_video_path) and delete_src_file and ori_video_path != save_video_path:
                os.remove(ori_video_path)#删掉源文件
                return
            if os.path.exists(save_video_path):#如果已经生成了，直接返回
                return
            probe = ffmpeg.probe(ori_video_path)
            video_info = next(c for c in probe["streams"] if c["codec_type"] == "video")
            width = video_info["width"]
            height = video_info["height"]
            
            resize_scale = shorter_size / float(min(width, height))
            resize_w = int(resize_scale * width) // 2 * 2
            resize_h = int(resize_scale * height) // 2 * 2
            stream = ffmpeg.input(ori_video_path)
            audio = stream.audio
            video = stream.video.filter("scale", resize_w, resize_h)
            ffmpeg.output(video, audio, save_video_path, r = fps).run()
            if os.path.exists(ori_video_path) and os.path.exists(save_video_path) and delete_src_file and ori_video_path != save_video_path:
                os.remove(ori_video_path)#删掉源文件
        except Exception as e:
            traceback.print_exc()
            return
    return
    num_process = 40

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
            if os.path.exists(feat_file):
                os.remove(feat_file)
            if os.path.exists(video_file):
                os.remove(video_file)
            #error_map[mp4] = 1
    print(good_txt)
    print(len(good_lst))
    write_file(good_lst, good_txt, split_str=" ")



def multiprocessing_ffmpeg_resize_refps(videos_path, dst_path, delete_src_file, num_process=20):
    videos_lst = get_file_list(videos_path, suffix=[".mp4"])
    # ffmpeg_mp4_lst(videos_lst, videos_path, dst_path, 0, delete_src_file)
    # ffmpeg_mp4_lst(videos_lst, videos_path, dst_path, 0)
    print("videos_lst", len(videos_lst))
    jpg_files = videos_lst
    import random
    random.shuffle(jpg_files)
    num_process = num_process
    fileArr = [[]for i in range(num_process)]
    for i in range(len(jpg_files)):
        fileArr[i % num_process].append(jpg_files[i])
    pool = multiprocessing.Pool()
    for j in range(num_process):
        pool.apply_async(ffmpeg_mp4_lst, args=(fileArr[j], videos_path, dst_path, j, delete_src_file))
    pool.close()
    pool.join()

def check_val_by_mp4_npy(val_txt, root_path, num_process=20):
    feat_path = os.path.join(root_path, "audio_features")
    video_path = os.path.join(root_path, "videos")
    val_lst = read_txt(val_txt, split_str=" ")
    print("val_lst", len(val_lst))
    file_lst = val_lst
    num_process = num_process
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
    print("src val_lst", len(val_lst))
    print(val_txt)
    #把good覆盖原来的

def convert_vid_url_to_caption_csv(dst_path, url_path, caption_path, cap_idx):

    print(url_path)
    feat_path = os.path.join(dst_path, "audio_features")
    vid_idx = 0
    url_idx = 1
    count = 0
    with open(caption_path, "w") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=["video_id", "merged_caption", "play_url"])
        writer.writeheader()
        reader = read_txt(url_path)
        print("reader", len(reader))
        for row in reader:
            if len(row) < cap_idx + 1:
                continue
            video_id = row[vid_idx]
            play_url = row[url_idx]
            caption = row[cap_idx]
            npy_name = video_id + ".npy"
            npy_file = os.path.join(feat_path, npy_name)
            if not os.path.exists(npy_file):
                continue
            if len(caption) == 0:
                continue
            count += 1
            res_dict = dict(video_id=video_id, merged_caption=caption, play_url=play_url)
            writer.writerow(dict(video_id=video_id, merged_caption=caption, play_url=play_url))
    print("caption count", count)
    print("args.caption_path")
    print("res_dict", res_dict)
    print(caption_path)

    
def move_video_audio_dst(src_lst, dst_path, root_path):
    for src_name in src_lst:
        src_path = os.path.join(root_path, src_name)
        #移动MP4
        src_videos_path = os.path.join(src_path, "videos")
        dst_videos_path = os.path.join(dst_path, "videos")
        print("dst_videos_path", dst_videos_path)
        src_videos = get_file_list(src_videos_path, [".mp4"])
        print("src_videos", src_videos_path, len(src_videos))
        count = 0
        for src_video in src_videos:
            dst_video = src_video.replace(src_path, dst_path)
            # print(src_video, dst_video)
            # copy_file(src_video, dst_video)
            move_file(src_video, dst_video)
            count += 1
            if count % 10000 == 0:
                print("video count", count)
            # break
        #移动feature
        src_videos_path = os.path.join(src_path, "audio_features")
        dst_videos_path = os.path.join(dst_path, "audio_features")
        src_videos = get_file_list(src_videos_path, [".npy"])
        print("audio_features", src_videos_path, len(src_videos))
        count = 0
        for src_video in src_videos:
            dst_video = src_video.replace(src_path, dst_path)
            # print(src_video, dst_video)
            # copy_file(src_video, dst_video)
            move_file(src_video, dst_video)
            count += 1
            if count % 10000 == 0:
                print("audio_features count", count)

def generate_sub_csv(val_txt, cap_csv, dst_cap_csv):
    val_lst = read_txt(val_txt, split_str=" ")
    cap_lst = read_caption_csv(cap_csv, keys=["video_id", "merged_caption", "play_url"])
    cap_map = {}
    for item in cap_lst:
        vid = item["video_id"]
        cap_map[vid] = item
    res_lst = []
    for item in val_lst:
        mp4 = item[0]
        vid = mp4[:-4]
        if cap_map.get(vid, None) is not None:
            res_lst.append(cap_map[vid])
    print("res_lst", len(res_lst))
    print(val_txt.split("/")[-1])
    print(dst_cap_csv.split("/")[-1])
    write_caption_csv(cap_csv=dst_cap_csv, line_dicts=res_lst, keys=["video_id", "merged_caption", "play_url"])

def split_train_val_by_val_num(total_txt, val_num):    
    src_lst = read_txt(total_txt)
    if args.val_num > 0:
        train_num = (len(src_lst) - val_num)/ float(len(src_lst))
    val_w = int(val_num/10000 + 1)
    train_w = int(train_num/10000 + 1)
    rand_lst = src_lst#进行shuffle
    random.shuffle(rand_lst)
    #生成0.7的训练集
    train_lst = rand_lst[:train_num]
    # train_lst = [l[1:] for l in train_lst]#去掉ID列名
    train_txt = total_txt + "_train{}".format(train_w)
    print("训练集路径", train_txt, "行数", len(train_lst), "列数", len(train_lst[0]))
    write_file(train_lst, train_txt, split_str=" ")
    #生成0.3的测试集
    test_lst = rand_lst[train_num:]
    # test_lst = [l[1:] for l in test_lst]#去掉ID列名
    test_txt = total_txt + "_val{}".format(val_w)
    print("测试集路径",  test_txt, "行数", len(test_lst),  "列数", len(test_lst[0]))#去掉ID列名
    write_file(test_lst, test_txt, split_str="\t")


if __name__ == '__main__':
    # id_label_map, label_id_map = txt_to_map("/home/daib/data/l1_l2_tags/87L2_label_index.txt")
    # assert(len(id_label_map.keys()) == len(label_id_map.keys()))
    # l1l2_lst = read_txt("/home/daib/data/l1_l2_tags/20L1_87L2.txt")
    # l1l2_map = lst_to_map(l1l2_lst, idx=0)
    # l2l1_map = lst_to_map(l1l2_lst, idx=1)
    
    # 新增数据到l2_train中
    # 第一步,新下载完数据之后,把其他数据移动到l2_train下
    # src_lst = ["20220911-0915"]
    # dst_path = "/home/daib/data/video_cls/l1_l2_tags/l2_train"
    # root_path = "/home/daib/data/video_cls/l1_l2_tags"
    # move_video_audio_dst(src_lst, dst_path, root_path)


    #第二步根据vid_url_l1_l2_cap,以及feat是否存在生成caption.total.csv
    #需要把下载的vid_url_l1_l2_cap.csv拷贝到l2_train目录下，然后和之前总的合并到一起
    #L2的cap_idx序号是4
    # dst_path = "/home/daib/data/l1_l2_tags/20221025_1026"
    # url_path = os.path.join(dst_path, "vid_url_l1_l2_cap_20221025_1026_5w.csv")
    # caption_path = os.path.join(dst_path, "caption.total_20221025_1026_5w.csv")
    # cap_idx = 4
    # # L3的cap_idx序号是5
    # convert_vid_url_to_caption_csv(dst_path, url_path, caption_path, cap_idx)


    # 第三步根据val_num将total_txt划分成_train和_val
    # total_txt = "/home/daib/data/l1_l2_tags/l2_train/l2_anno.txt.total_20220930_1004_27w.txt"
    # val_num = 10000
    # split_train_val_by_val_num(total_txt, val_num)


    # 第四步根据根据val和总的caption.total.csv生成子sub.caption.csv
    #cap_csv基本不用修改
    # cap_csv = "/home/daib/data/l1_l2_tags/l2_train/caption_ocr.total_20220802-1024_425w.csv"
    # val_txt = "/home/daib/data/l1_l2_tags/l2_train/l2_anno.txt.total_20220802-1024_425w.txt_6.4.8_thd0.2_6.6.4_thd0.7.txt_merge398w.txt"
    # dst_cap_csv = cap_csv + ".sub_6.4.8_thd0.2_6.6.4_thd0.7.txt_merge398w.csv"
    # generate_sub_csv(val_txt, cap_csv, dst_cap_csv)

    #第五步可以将两个caption.total.csv进行合并


    # #第五步多进程检查训练样本是否合法, 在下载的时候已经检查过了, 有需要还可以再检查一次
    root_path = "/home/liwang/data/l3_tag/l3_0929_66w/"
    val_txt = os.path.join(root_path, "l3_anno.txt.total_0929_66w.train62w")
    check_val_by_mp4_npy(val_txt, root_path, num_process=48)
    

    #第六步在下载的时候，应该就会根据多线程检查，生成合格的val,把mp4用ffmpeg下采样5帧，压缩空间
    # videos_path = "/home/daib/data/video_cls/l1_l2_tags/20220911-0915/videos"
    # dst_path = videos_path + "_r5"
    # delete_src_file = True
    # multiprocessing_ffmpeg_resize_refps(videos_path, dst_path, delete_src_file, num_process=24)
