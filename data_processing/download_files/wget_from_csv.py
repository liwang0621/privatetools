import os
import sys
import numpy as np
from tqdm import tqdm
import pandas as pd
import multiprocessing
from multiprocessing import Manager

from loguru import logger

# data = pd.read_csv('/home/liwang/data/shopee_shoes/spu_comment/spu_comment_sample_shoes_all.csv', delimiter="\t")

# all_urls = []
# for urls in data['video_url']:
#     for url in urls.split(','):
#         all_urls.append(url)
all_urls = []
with open('/home/liwang/data/product_data/检测训练数据/train.txt', 'r') as f:
    for line in f.readlines():
        url = line.split()[0]
        all_urls.append(url)

from IPython import embed; embed()


# all_urls = all_urls[60000:]

# from pdb import set_trace; set_trace()

save_root = '/home/liwang/data/product_data/检测训练数据/images'
os.makedirs(save_root, exist_ok=True)

done_urls = []

def worker(procnum, urls, save_root):
    logger.info('processing {}'.format(procnum))
    for url in tqdm(urls):
        name = os.path.basename(url)
        # name = name + '.jpg'
        # cmd = 'wget -P {}/images/ {}'.format(save_root, url)
        save_path = '{}/{}.jpeg'.format(save_root, name)
        if url not in done_urls:
            while os.path.exists(save_path):
                save_path = save_path.replace('.jpeg', '1.jpeg')
            cmd = 'wget -O {} -c {}'.format(save_path, url)
            os.system(cmd)
            done_urls.append(url)
        # print(cmd)



multithread = 64

nr_lines = len(all_urls)
nr_liner_per_worker = np.ceil(nr_lines / multithread)


manager = Manager()
# return_list = manager.list() 也可以使用列表list
return_dict = manager.dict()
jobs = []
for i in range(multithread):
    device = 'cuda:{}'.format(i%8)

    start = int(i * nr_liner_per_worker)                                               
    end = int(min(start + nr_liner_per_worker, nr_lines))
    image_paths_worker = all_urls[start:end]

    p = multiprocessing.Process(target=worker, args=(i, image_paths_worker, save_root))
    jobs.append(p)
    p.start()

for proc in jobs:
    proc.join()
