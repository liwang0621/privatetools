import librosa
import multiprocessing
import os
import numpy as np


def extract_feature(wav_paths, save_dir, sr=16000, win_size=128, step_size=64, n_mels=80):
    for i, wav_path in enumerate(wav_paths):
        try:
            if i % 1000 == 0:
                print("extract_feature count", i, len(wav_paths))
            base_name = os.path.basename(wav_path).split(".")[0]
            save_path = os.path.join(save_dir, base_name + ".npy")
            if os.path.exists(save_path):
                continue
            y, _ = librosa.load(wav_path, sr=sr)
            n_fft = sr * win_size // 1000
            hop_length = sr * step_size // 1000
            mel = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            log_mel = librosa.power_to_db(mel)
            shape = log_mel.shape
            shape0 = shape[0]
            with open(save_path, "wb") as f:
                np.save(f, log_mel)
            if os.path.exists(save_path) and os.path.exists(wav_path):#如果转成logmel之后，直接删掉
                os.remove(wav_path)
        except Exception as e:
            continue


def run(wav_paths, save_dir, sr, win_size, step_size, n_mels, process_num=1):
    process_num = min(process_num, len(wav_paths))
    step = len(wav_paths) // process_num
    dispatch_wav_paths = []
    for process_idx in range(0, process_num):
        start_idx = process_idx * step
        if process_idx == process_num - 1:
            end_idx = len(wav_paths)
        else:
            end_idx = (process_idx + 1) * step
        dispatch_wav_paths.append(wav_paths[start_idx:end_idx])
    processes = []
    for process_idx in range(0, process_num):
        print("process {} gets {} items".format(process_idx, len(dispatch_wav_paths[process_idx])))
        process = multiprocessing.Process(target=extract_feature, args=[dispatch_wav_paths[process_idx],
                                                                        save_dir, sr, win_size, step_size, n_mels])
        process.daemon = True
        processes.append(process)
    for process in processes:
        process.start()
    for process in processes:
        process.join()


def load_wav_paths(curr_path):
    if curr_path is None:
        return []
    elif os.path.isfile(curr_path):
        if curr_path.endswith(".wav"):
            return [curr_path]
        else:
            return []
    else:
        total_res = []
        for sub_name in os.listdir(curr_path):
            sub_path = os.path.join(curr_path, sub_name)
            total_res.extend(load_wav_paths(sub_path))
        return total_res

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
    
    #设置audios路径
    # wav_dir = "/home/daib/data/video_cls/l1_l2_tags/20220805_20w/audios"


    save_dir = os.path.join(os.path.dirname(args.audios_path), "audio_features")
    sr = 16000
    win_size = 128
    step_size = 64
    n_mels = 80
    process_num = 48

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    total_wav_paths = load_wav_paths(args.audios_path)
    run(total_wav_paths, save_dir, sr, win_size, step_size, n_mels, process_num)
