import subprocess
import os
import multiprocessing
import tqdm
import traceback
def extract_wav(video_names, ori_dir, save_dir, audio_features_path):
    for video_name in tqdm.tqdm(video_names):
        try:
            ori_path = os.path.join(ori_dir, video_name)
            if not os.path.exists(ori_path):
                print("WARNING mp4 not exist", ori_path)
                continue
            vid = video_name[:-4]
            npy_name = vid + '.npy'
            feat_file = os.path.join(audio_features_path, npy_name)
            if os.path.exists(feat_file):
                print("WARNING feat_file  exist", feat_file)
                continue
            video_banse_name = os.path.basename(video_name).split(".")[0]
            print("video_banse_name", video_banse_name)
            save_path = os.path.join(save_dir, "{}.wav".format(video_banse_name))
            print("save_path", save_path)
            cmd = "ffmpeg -i {} -f wav -ar 16000 {}".format(ori_path, save_path)
            # "ffmpeg -i {} -vn -ar 44100 -ac 2 -ab 96k {}".format(ori_path, save_path)
            print("cmd", cmd)
            if os.path.exists(save_path):
                print("WARNING wav exist!!", save_path)
                continue
            status_code, _ = subprocess.getstatusoutput(cmd)
            if status_code != 0 and os.path.exists(save_path):
                os.remove(save_path)
        except Exception as e:
            traceback.print_exc()
            print(e)


def run(ori_video_names, ori_dir, save_dir, audio_features_path, process_num=1):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    process_num = min(process_num, len(ori_video_names))
    step_size = len(ori_video_names) // process_num
    dispatch_video_names = []
    for process_idx in range(0, process_num):
        start_idx = process_idx * step_size
        if process_idx == process_num - 1:
            end_idx = len(ori_video_names)
        else:
            end_idx = (process_idx + 1) * step_size
        dispatch_video_names.append(ori_video_names[start_idx:end_idx])

    processes = []
    for process_idx in range(0, process_num):
        print("process {} gets {} items".format(process_idx, len(dispatch_video_names[process_idx])))
        process = multiprocessing.Process(target=extract_wav, args=[dispatch_video_names[process_idx], ori_dir, save_dir, audio_features_path])
        process.daemon = True
        processes.append(process)
    for process in processes:
        process.start()
    for process in processes:
        process.join()


def load_videos(anno_path):
    video_names = []
    with open(anno_path, "r") as f:
        for line in f.readlines():
            video_name = line.strip().split("\t")[0] + ".mp4"
            video_names.append(video_name)
    return video_names
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
import argparse
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
    # args.root_path = "/home/daib/data/video_cls/l1_l2_tags/20220805_20w "
    
    args.videos_path = os.path.join(args.root_path, "videos")
    args.audios_path = os.path.join(args.root_path, "audios")
    args.audio_features_path = os.path.join(args.root_path, "audio_features")
    
    # anno_path = "/home/daib/data/video_cls/l1_l2_tags/20220805_20w/vid_url_l1_l2_cap_20220802_0808_35w.csv"#\t隔开，第一列是vid，第二列是url
    # ori_video_dir = "/home/daib/data/video_cls/l1_l2_tags/20220805_20w/videos"
    # save_audio_dir = "/home/daib/data/video_cls/l1_l2_tags/20220805_20w/audios"

    video_names = load_videos(args.url_path)
    run(video_names,  args.videos_path, args.audios_path, args.audio_features_path, args.process_num)

    #kinetics400
    # anno_path = "/home/daib/data/video_cls/kinetics400/k400/annotations/val.csv.exist"#\t隔开，第一列是vid，第二列是url
    # ori_video_dir = "/home/daib/data/video_cls/kinetics400/k400_targz/val/videos"
    # save_audio_dir = "/home/daib/data/video_cls/kinetics400/k400_targz/val/audios"
    # process_num = 48
    # anno_lst = read_txt(anno_path,split_str=" ")
    # video_names = [l[0] for l in anno_lst]

    # run(video_names, ori_video_dir, save_audio_dir, process_num)
