import cv2
import os
import glob
import copy
from multiprocessing import Pool
import random
import json

def get_filelist(dir, Filelist, search_exts= ['.mp4', '.MOV']):
    newDir = dir
    if os.path.isfile(dir) and os.path.splitext(dir)[-1] in search_exts:
        Filelist.append(dir)
        # # 若只是要返回文件文，使用这个
        # Filelist.append(os.path.basename(dir))
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # 如果需要忽略某些文件夹，使用以下代码
            #if s == "xxx":
                #continue
            newDir=os.path.join(dir,s)
            get_filelist(newDir, Filelist)
    return Filelist

#差值感知算法
def dHash(img):
    #缩放8*8
    img=cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    #转换灰度图
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str=''
    #每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if   gray[i,j]>gray[i,j+1]:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str

#Hash值对比
def cmpHash(hash1,hash2):
    n=0
    #hash长度不同则返回-1代表传参出错
    if len(hash1)!=len(hash2):
        return -1
    #遍历判断
    for i in range(len(hash1)):
        #不相等则n计数+1，n最终为相似度
        if hash1[i]!=hash2[i]:
            n=n+1
    return n

def get_min_cmpHash(hash_str, hashs_list):
    min_diff = 100
    for s_hash in hashs_list:
        diff = cmpHash(hash_str, s_hash)
        if diff < min_diff:
            min_diff = diff
    return min_diff
 
# video_paths = glob.glob('/home/liwang/data/shopee_shoes/spu_comment/videos_v2/*.mp4') #填入自己对应视频的地址
Filelist = []
video_paths = get_filelist('/home/liwang/data/face_mask/videos/buy_2022-01-25', Filelist)

print("total video:", len(video_paths))
# video_paths = video_paths[:100000]
print("total video:", len(video_paths))
frame_save_dir = '/home/liwang/data/face_mask/images/20220125' #对应抽帧输出目录
if not os.path.exists(frame_save_dir):
    os.makedirs(frame_save_dir)

gap = 30 #抽帧间隔
num_process = 32 #进程数量
 
# import IPython
# IPython.embed()
# random.shuffle(video_paths)
# gap = 15
 
 
# def process_video2frame(part_video_list):
#     cnt = 0
#     for idx, video_path in enumerate(part_video_list):
#         v_name = os.path.basename(video_path)
#         v_name = v_name[:v_name.rfind('.')]
#         video_frame_save_dir = os.path.join(frame_save_dir, v_name)
#         if not os.path.exists(video_frame_save_dir):
#             os.makedirs(video_frame_save_dir)
#         else:
#             print("already exist, ", video_frame_save_dir)
#             if os.path.exists(os.path.join(video_frame_save_dir, '00000.jpg')):
#                 continue
#             # continue
 
#         frame_cnt = 0
#         cap = cv2.VideoCapture(video_path)
#         success, frame = cap.read()
#         while success:
#             # frame = cv2.resize(frame, (171, 128))
#             if frame_cnt % gap == 0:
#                 cv2.imwrite(os.path.join(video_frame_save_dir, '{:0>5d}.jpg'.format(frame_cnt)), frame)
#             frame_cnt += 1
#             success, frame = cap.read()
#         cap.release()
 
#         print("{}: {} done.".format(idx + 1, v_name))

def process_video2frame(part_video_list):
    cnt = 0
    for idx, video_path in enumerate(part_video_list):
        # v_name = os.path.basename(video_path)
        # v_name = v_name[:v_name.rfind('.')]
        # v_name = v_name.replace('/', '_')
        v_name = os.path.splitext(video_path)[0].split('/home/liwang/data/face_mask/videos/buy_2022-01-25/')[-1]
        v_name = v_name.replace('/', '_')

        if os.path.exists(os.path.join(frame_save_dir, v_name + '_' + '00000.jpg')):
            continue
        # continue
 
        frame_cnt = 0
        cap = cv2.VideoCapture(video_path)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print('frame_count: {}'.format(frame_count))
        # frame_count = 1000
        success, frame = cap.read()
        while success:
            # frame = cv2.resize(frame, (171, 128))
            if gap < 0:
                if frame_cnt == int(frame_count * 0.1) or frame_cnt == int(frame_count * 0.6):
                    frame = cv2.flip(frame, 0)
                    cv2.imwrite(os.path.join(frame_save_dir,  v_name + '_' + '{:0>5d}.jpg'.format(frame_cnt)), frame)
            else:
                if frame_cnt % gap == 0:
                    frame = cv2.flip(frame, 0)
                    cv2.imwrite(os.path.join(frame_save_dir,  v_name + '_' + '{:0>5d}.jpg'.format(frame_cnt)), frame)
            frame_cnt += 1
            success, frame = cap.read()
        cap.release()
 
        print("{}: {} done.".format(idx + 1, v_name))

def process_video2frame_diff(part_video_list):
    cnt = 0
    for idx, video_path in enumerate(part_video_list):
        # v_name = os.path.basename(video_path)
        # v_name = v_name[:v_name.rfind('.')]
        # v_name = v_name.replace('/', '_')
        v_name = os.path.splitext(video_path)[0].split('/home/liwang/data/face_mask/videos/buy_2022-01-25/')[-1]
        v_name = v_name.replace('/', '_')

        if os.path.exists(os.path.join(frame_save_dir, v_name + '_' + '00000.jpg')):
            continue
        # continue
 
        frame_cnt = 0
        cap = cv2.VideoCapture(video_path)
        # frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # print('frame_count: {}'.format(frame_count))
        # frame_count = 1000
        success, frame = cap.read()
        # if '0001/1平视/1-2.mp4' in video_path:
        #     from IPython import embed; embed()
        tag = copy.deepcopy(success)
        skip_num = random.randint(1, gap)
        dhashs = []
        while success:
            # frame = cv2.resize(frame, (171, 128))
            if frame_cnt > skip_num:
                if frame_cnt % gap == 0:
                    frame = cv2.flip(frame, 0)
                    dhash_str= dHash(frame)
                    if len(dhashs) == 0:
                        dhashs.append(dhash_str)
                        cv2.imwrite(os.path.join(frame_save_dir,  v_name + '_' + '{:0>5d}.jpg'.format(frame_cnt)), frame)
                    elif get_min_cmpHash(dhash_str, dhashs) >= 20:
                        dhashs.append(dhash_str)
                        # print('saved in {}'.format(os.path.join(frame_save_dir,  v_name + '_' + '{:0>5d}.jpg'.format(frame_cnt))))
                        cv2.imwrite(os.path.join(frame_save_dir,  v_name + '_' + '{:0>5d}.jpg'.format(frame_cnt)), frame)
            frame_cnt += 1
            success, frame = cap.read()
        cap.release()
 
        print("{}: {} - {} - {} done.".format(idx + 1, tag, v_name, frame_cnt))
 
 
print('Parent process %s.' % os.getpid())
p = Pool(num_process)
part_list_len = len(video_paths) // num_process
for i in range(num_process):
    if i == num_process - 1:
        part_list = video_paths[part_list_len * i:]
    else:
        part_list = video_paths[part_list_len * i:part_list_len * (i + 1)]

    print(part_list)
    p.apply_async(process_video2frame_diff, args=(part_list,))
print('Waiting for all subprocesses done...')
p.close()
p.join()
print('All subprocesses done.')