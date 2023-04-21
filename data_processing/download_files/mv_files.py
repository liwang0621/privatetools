import os
from tqdm import tqdm 

source_dir = '/home/liwang/data/shopee_shoes/spu_comment/video_10w_cls_res/shoes_with_leg'
target_dir = '/home/liwang/data/shopee_shoes/spu_comment/video_10w_cls_res/shoes_with_leg_10w'

file_list = os.listdir(source_dir)
file_list = file_list[:100000]

for name in tqdm(file_list):
    cmd = 'mv {}/{} {}/'.format(source_dir, name, target_dir)
    print(cmd)
    os.system(cmd)