#!/usr/bin/env bash
export OMP_NUM_THREADS=1

val_num=40000 #为0就按照0.05val取
# root_path="/home/daib/data/l1_l2_tags/test_3.8w"
# url_path="/home/daib/data/l1_l2_tags/test_3.8w/vid_url_l1_l2_cap_test_3.8w_val_4w.csv"
# date="test_3.8w_val_4w" #后面的命名将根据这个进行命名
root_path="/home/liwang/data/l3_tag/l3_1109_100w"
url_path="/home/liwang/data/l3_tag/l3_1109_100w/content_profile_20221109-125207.csv"
date="20221109-125207" #后面的命名将根据这个进行命名
cap_idx=5 #l2是4，l3是5

# # 下载视频
python step1_download_videos.py --url_path=${url_path} --root_path=${root_path} &
wait
# python step1_download_videos.py --url_path=${url_path} --root_path=${root_path} &
# wait 

# # #抽取视频
# python step2_extract_audios.py --url_path=${url_path} --root_path=${root_path} &
# wait

# # #抽取audio_feature
# python step3_extract_audio_features.py --url_path=${url_path} --root_path=${root_path} &
# wait

# # # # # #生成验证集，并通过多线程检查npy是否存在,注意l2和l3的caption的位置不一样
# # # #再合到l2_anno.txt.train
# python step5_2_l2_gen_val.py --url_path=${url_path} --root_path=${root_path} --date=${date}&
# wait


# #备份
# # #提取caption,后面每次把新增的vid_url合并到老的vid_url之后， 再进行生成caption_total 
# # python step4_extrac_caption.py --url_path=${url_path} --root_path=${root_path} --cap_idx=${cap_idx} --date=${date} &
# # wait

# # #进行验证集分割
# # python step5_1_split_train_val.py  --url_path=${url_path} --root_path=${root_path} --val_num=${val_num} &
# # wait
