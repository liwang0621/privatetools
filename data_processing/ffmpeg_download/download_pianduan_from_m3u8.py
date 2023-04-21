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
import sys
import multiprocessing
import json
import shutil
import random
import argparse
import numpy as np
import pandas as pd
import traceback
import csv


import base64
import hmac
import os
import requests
import hashlib
import json



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
            print("ERROR:dst_file exists!", dst_file)
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
            print("ERROR:dst_file exists!", dst_file)
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
    try:  # 如果是标签id，进行排序后输出
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
parser.add_argument('--dst_root', default="", type=str, help='')
args = parser.parse_args()


def print_index_columns(data, sign=""):
    indexs = data._stat_axis.values.tolist()  # 行名称
    columns = data.columns.values.tolist()  # 列名
    print("文件名", sign, "行索引数", len(indexs), "前2行", indexs[:2], "最后2行", indexs[-2:])
    print("文件名", sign, "列索引数", len(columns), "前2列", columns[:2], "最后2列", columns[-2:])
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


def write_df(df, csv_file="tmp.csv"):
    indexs, colums = print_index_columns(df)
    data = df.values.tolist()
    res_lst = []
    res_lst.append(["ID"] + colums)
    assert (len(data) == len(indexs))
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

def get_num_ratio_singlelab(val_txt):
    """获取单标签文件的分布，"L2labid", "L2lab", "num", "ratio(%)"
    """
    val_lst = read_txt(val_txt, split_str=" ")
    val_map = lst_to_map(val_lst, idx=1)#如果是mp4_vid格式，idx=1,
    ##如果是vid_gt_pred_score格式，idx=2
    print("val_lst", val_map.keys())
    res_lst = []
    sum = len(val_lst)
    for labid in id_label_map:
        lab = id_label_map[labid]
        if val_map.get(labid, None) is None:
            num = 0
        else:
            num = len(val_map[labid])
        ratio = round(float(num)*100/sum, 2)
        res_lst.append([labid, id_label_map[str(labid)],num, ratio])
    res_lst.insert(0, ["L2labid", "L2lab", "num", "ratio(%)"])
    res_txt = val_txt + "_labid_num.txt"
    print("res_txt", res_txt)
    print("res_lst", len(res_lst))
    print(res_lst)
    write_file(res_lst, res_txt)


def filter_val_lst(val_lst):
    #前两个都是数字,就是正确的行，否则把这一行的第一个元素，加上good_i的最后一个元素，然后其他还有其他的话，也一起拼接过去
    good_i = 0
    for i, item in enumerate(val_lst):
        val_lst[i] = [l.strip('"') for l in item]
    for i, item in enumerate(val_lst):
        if len(item) >= 2 and item[0].isdigit() and item[1].isdigit():
            good_i = i
            continue
        else:
            val_lst[good_i][-1] += item[0]
            if len(item) > 1:
                val_lst[good_i].extend(item[1:])
    res_lst = []
    for i, item in enumerate(val_lst):
        if len(item) >= 2 and item[0].isdigit() and item[1].isdigit():
            if len(item) != len(val_lst[0]):
                print(i)
                continue
            res_lst.append(item)
    return res_lst


def read_caption_csv(cap_csv, keys=[]):
    """通过key读取dict
    """
    res_lst = []
    print("read_caption_csv", keys)
    with open(cap_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            line_dict = dict()
            if len(keys) == 0:
                for key in row:
                    keys.append(key)
            for key in keys:
                line_dict[key] = row[key]
            res_lst.append(line_dict)
    return res_lst


def write_caption_csv(cap_csv=None, line_dicts=None, keys=None):
    if keys is None:
        keys = list(line_dicts[0].keys())
    with open(cap_csv, "w") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=keys)
        writer.writeheader()
        for row in line_dicts:
            write_dict = dict()
            for key in keys:
                write_dict[key] = row[key]
            writer.writerow(write_dict)

def hash_hmac(key, msg):
    hmac_code = hmac.new(key.encode(), msg.encode(), hashlib.sha1).digest()
    return base64.b64encode(hmac_code).decode('ascii')


secret_id_and_key = {
    'live': {
        "rcmd": "RiLFjRkV1zW4PPoGUhISXk2jRI7I5BScDdXKNmgyT8gHfg2vSI9uWq3PPDkyd9fx"
    },
    'uat': {
        "rcmd": "aA2wHFl7OTolefR24CApZoJEhUSDA8Eas2w5B9O27IRJqZZhzroqX0rIsEQHb8TI"
    },
    'test': {
        "rcmd": "kVRZj9GHpVEWz4MFvBZqQmZkaD9mCDJJibXCHc4Xp4xFDqRRjSOWDwc5fl0NhFyj"
    }
}

region_mapping = {
    "ID":"live-record.livetech.shopee.co.id",
    "TH":"live-record.livetech.shopee.co.th",
    "MY":"live-record.livetech.shopee.com.my",
    "BR":"live-record.livetech.shopee.com.br",
    "SG":"live-record.livetech.shopee.sg",
    "TW":"live-record.livetech.shopee.tw",
    "VN":"live-record.livetech.shopee.vn",
    "PH":"live-record.livetech.shopee.ph",
}


def get_u3m8(region, session_id):
    data = {
        "SecretId": "rcmd",
        "X-Shopee-Nonce": "10034",                # random
        "X-Shopee-Term": "1971285109",            # unix timestamp
        "app_id": "LS",
        "session_id": "LS:{}".format(session_id),
    }

    key = secret_id_and_key['live']['rcmd']
    msg = '&'.join([f'{k}={v}' for k, v in data.items()])
    print(key, msg)

    signature = hash_hmac(key, msg)
    print(signature)

    site = region_mapping[region]
    url = 'https://{}/api/v1/record/url?app_id={}&session_id={}'.format(site, data['app_id'], data['session_id'])
    headers = {
        'Authorization': "SP-HMAC-SHA1 SecretId=rcmd, Signature={}".format(signature),
        'Content-Type': "application/json; charset=utf-8",
        'X-Shopee-Nonce': data["X-Shopee-Nonce"],
        "X-Shopee-Term": data["X-Shopee-Term"],
    }
    print(url)
    print(headers)

    r = requests.get(url, headers=headers)
    print(r.text)
    return r.text
    #cmd = "curl -L -X POST 'https://{}/api/v1/record/url?app_id={}&session_id={}' -H 'Authorization: SP-HMAC-SHA1 SecretId=rcmd, Signature={}' -H 'X-Shopee-Nonce: {}' -H 'X-Shopee-Term: {}'".format(
    #    site, data['app_id'], data['session_id'],
    #    signature, data['X-Shopee-Nonce'], data['X-Shopee-Term']
    #)
    #print(cmd)
    #os.system(cmd)

# get_u3m8('VN', 3559146)
# print()

import requests
def get_record_url(grass_region, seesion_id):
    record_urls = ""
    try:
        response_text = get_u3m8(grass_region, seesion_id)
        res_dict = json.loads(response_text)
        tmp = res_dict["data"]["record_urls"]
        record_urls = "___".join(list(set(res_dict["data"]["record_urls"])))
    except:
        traceback.print_exc()
        print("session_id {} no record_url".format(seesion_id))
        pass
    return record_urls


if __name__ == '__main__':
    grass_region = 'VN'

    save_root = '/home/liwang/data/live_tag_data/tag_mining/videos/good'
    with open('/home/liwang/data/live_tag_data/tag_mining/gmv_id_good.txt', 'r') as f:
        for line in f.readlines()[:1]:
            tmp = line.split()
            gmv = float(tmp[0])
            session_id = str(tmp[1])
            # session_id = 3456180
            record_url = get_record_url(grass_region, session_id)

            start_time = random.randint(1,600)

            cmd = 'ffmpeg -i {} -ss {} -t 60 -loglevel fatal -nostdin -y -c copy {}/{}.mp4'.format(record_url, start_time, save_root, session_id)
            print(cmd)
            # os.system(cmd)