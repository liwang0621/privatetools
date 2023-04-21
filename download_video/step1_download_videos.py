#coding=utf-8
import argparse
import multiprocessing
import os
import urllib.request
import shutil
import io
import pandas as pd
from mmcv.fileio import FileClient
import traceback
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# def load_urls(url_path):
#     urls = []
#     vid_set = set()
#     with open(url_path, "r", encoding='utf-8') as f:
#         for line in f.readlines():
#             info = line.strip().split("\t")
#             if len(info) < 2:
#                 continue
#             # print("info", info)
#             vid = info[0]
#             url = info[1]
#             if not vid in vid_set:
#                 vid_set.add(vid)
#                 urls.append((vid, url))
#     from IPython import embed; embed()
#     return urls

def load_urls(url_path):
    urls = []
    vid_set = set()
    data = pd.read_csv(url_path)
    for index, it in data.iterrows():
        vid = str(it['video_id'])
        url = it['play_url']
        if not vid in vid_set:
            vid_set.add(vid)
            urls.append((vid, url))
    return urls[:500000]


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

import ffmpeg
def ffmpeg_resize_refps(ori_video_path, save_video_path, shorter_size=256, fps=5, delete_src_file=True):
    if shorter_size <= 0:#不进行resize
        os.rename(ori_video_path, save_video_path)
    else:
        try:
            if os.path.exists(ori_video_path) and os.path.exists(save_video_path) and delete_src_file and ori_video_path != save_video_path:
                os.remove(ori_video_path)#如果已经生成过了删掉源文件
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
            if os.path.exists(ori_video_path):#如果不能进行ffmpeg
                os.remove(ori_video_path)#删掉源文件
            return
    return

def download(urls, save_dir, shorter_size, process_idx):
    print("download(urls, save_dir, shorter_size, process_idx)", process_idx)
    file_client = FileClient('disk')
    import time
    s_t = time.time()
    for i, (vid, url) in enumerate(urls):
        import ffmpeg
        if i % 1000 == 0:
            print("count", i, "total", len(urls), "need time(h)", (time.time() - s_t)*(len(urls) - i)/ (3600 * (len(urls) + 1)))
        ori_video_path = os.path.join(save_dir, vid + "_ori.mp4")
        save_video_path = os.path.join(save_dir, vid + ".mp4")
        if os.path.isfile(save_video_path):
            print("WARNING!!!exist file", save_video_path)
            continue
        try:
            # print("urllib.request.urlopen", url)
            data = urllib.request.urlopen(url,timeout=1).read()
            with open(ori_video_path, "wb") as f:
                f.write(data)
            # print("time_cost download", process_idx, i, time.time() - s_t)
            s_t = time.time()

            # print("success download {}".format(url))
            # print("save_video_path", save_video_path)

            # ffmpeg_mp4_r5(ori_video_path, save_video_path, delete_src_file=True)
            ffmpeg_resize_refps(ori_video_path, save_video_path, shorter_size=256, fps=5, delete_src_file=True)
            # if shorter_size <= 0:
            #     os.rename(ori_video_path, save_video_path)
            # else:
            #     probe = ffmpeg.probe(ori_video_path)
            #     video_info = next(c for c in probe["streams"] if c["codec_type"] == "video")
            #     width = video_info["width"]
            #     height = video_info["height"]
                
            #     resize_scale = shorter_size / float(min(width, height))
            #     resize_w = int(resize_scale * width) // 2 * 2
            #     resize_h = int(resize_scale * height) // 2 * 2
            #     print("vid width, height，resize_w, resize_h", vid, width, height, resize_w, resize_h)
            #     stream = ffmpeg.input(ori_video_path)
            #     audio = stream.audio
            #     video = stream.video.filter("scale", resize_w, resize_h)
            #     try:
            #         ffmpeg.output(video, audio, save_video_path).run()
            #     except Exception as e:
            #         continue
            # # file_obj = io.BytesIO(file_client.get(save_video_path))
            # if os.path.isfile(ori_video_path):
            #     #print("os.path.isfile(ori_video_path)", ori_video_path)
            #     os.remove(ori_video_path)
            # print("time_cost ffmpeg", process_idx, i, time.time() - s_t)
        except Exception as e:
            traceback.print_exc()
            print("ERROR")
            # if os.path.isfile(save_video_path):
            #     os.remove(save_video_path)
            #     print("os.path.isfile(save_video_path)", save_video_path)
                
        


def check_exists(urls, save_dir):
    res_urls = []
    for item in urls:
        vid = item[0]
        save_video_path = os.path.join(save_dir, vid + ".mp4")
        if not os.path.exists(save_video_path):
            res_urls.append(item)
    return res_urls

def run(url_path, save_dir, shorter_size, process_num):
    if os.path.isdir(save_dir):
        print("os.path.isdir(save_dir)", save_dir)
        #shutil.rmtree(save_dir)
    else:
        os.makedirs(save_dir)
    print("shorter_size", shorter_size)
    print("process_num", process_num)
    urls = load_urls(url_path)  
    urls = check_exists(urls, save_dir)  
    print("urls", len(urls))
    if len(urls) == 0:
        return
    # download(urls, save_dir, shorter_size, 0)
    # return
    import random
    random.shuffle(urls)
    process_num = min(process_num, len(urls))
    step_size = len(urls) // process_num
    processes = []
    for process_idx in range(0, process_num):
        start_idx = process_idx * step_size
        end_idx = (process_idx + 1) * step_size
        if process_idx == process_num - 1:
            end_idx = len(urls)
        sub_urls = urls[start_idx:end_idx]
        process = multiprocessing.Process(target=download, args=[sub_urls, save_dir, shorter_size, process_idx])
        process.daemon = True
        processes.append(process)
    for process in processes:
        process.start()
    for process in processes:
        process.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url_path", type=str, default="",help="path of url file")
    parser.add_argument("--root_path", type=str, default="",help="path of url file")
    parser.add_argument("--videos_path", type=str, default="",help="path of url file")
    parser.add_argument("--audios_path", type=str, default="",help="path of url file")
    parser.add_argument("--audio_features_path", type=str, default="",help="path of url file")

    parser.add_argument("--shorter_size", type=int, default=256)
    parser.add_argument("--process_num", type=int, default=48)

    args = parser.parse_args()
    # args.url_path = "/home/daib/data/video_cls/l1_l2_tags/20220805_20w/vid_url_l1_l2_cap_20220802_0808_35w.csv"#\t隔开，第一列是vid，第二列是url
    # args.root_path = "/home/daib/data/video_cls/l1_l2_tags/20220805_20w"
    
    args.videos_path = os.path.join(args.root_path, "videos")
    args.audios_path = os.path.join(args.root_path, "audios")
    args.audio_features_path = os.path.join(args.root_path, "audio_features")
    
    #设置url.csv

    # args.url_path = "/home/daib/data/video_cls/l1_l2_tags/20220805_20w/vid_url_l1_l2_cap_20220802_0808_35w.csv"#\t隔开，第一列是vid，第二列是url
    # #输出
    # save_dir = "/home/daib/data/video_cls/l1_l2_tags/20220805_20w/videos"#数据集路径下面加videos

    # dataset = os.path.basename(args.url_path).split(".")[0]
    # dataset = os.path.join(dataset, "videos")
    # save_dir = os.path.join(save_root, "videos")
    print("args.url_path", args.url_path)
    run(args.url_path, args.videos_path, args.shorter_size, args.process_num)

