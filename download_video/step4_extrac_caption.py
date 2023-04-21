import os
import csv
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
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url_path", type=str, default="",help="path of url file")
    parser.add_argument("--root_path", type=str, default="",help="path of url file")
    parser.add_argument("--videos_path", type=str, default="",help="path of url file")
    parser.add_argument("--audios_path", type=str, default="",help="path of url file")
    parser.add_argument("--audio_features_path", type=str, default="",help="path of url file")
    parser.add_argument("--caption_path", type=str, default="",help="path of url file")

    parser.add_argument("--shorter_size", type=int, default=256)
    parser.add_argument("--process_num", type=int, default=48)
    parser.add_argument("--vid_idx", type=int, default=0)
    parser.add_argument("--url_idx", type=int, default=1)
    parser.add_argument("--cap_idx", type=int, default=4)
    parser.add_argument("--date", type=str, default="",help="")
    args = parser.parse_args()
    args.videos_path = os.path.join(args.root_path, "videos")
    args.audios_path = os.path.join(args.root_path, "audios")
    args.audio_features_path = os.path.join(args.root_path, "audio_features")
    args.caption_path = os.path.join(args.root_path, "caption.total_{}.csv".format(args.date))
    #生成字典格式，训练用到，video_id和merged_caption
    # url_path = "/home/daib/data/video_cls/l1_l2_tags/20220805_20w/20220805_vid_url_l1_l2_cap.csv"
    # vid_idx = 0
    # url_idx = 1
    # cap_idx = 4
    # caption_path = "/home/daib/data/video_cls/l1_l2_tags/20220805_20w/caption.total.csv"
    count = 0
    with open(args.caption_path, "w") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=["video_id", "merged_caption", "play_url"])
        writer.writeheader()
        reader = read_txt(args.url_path)
        for row in reader:
            if len(row) < args.cap_idx + 1:
                print("ERROR len(row) < cap_idx + 1", row)
                continue
            video_id = row[args.vid_idx]
            play_url = row[args.url_idx]
            caption = row[args.cap_idx]
            if len(play_url) == 0:
                print("play_url empty", row)
                continue
            if len(caption) == 0:
                print("caption empty", row)
                continue
            count += 1
            writer.writerow(dict(video_id=video_id, merged_caption=caption, play_url=play_url))
    print("caption count", count)
    print("args.caption_path")
    print(args.caption_path)