import os
from glob import glob
from tqdm import tqdm
import sys
sys.path.append('~/projects/mmclassification/')
from mmcls.apis import inference_model, init_model, show_result_pyplot

def annostr2dict(annostr):
    info = annostr.strip().split()
    img_name = info[0]
    img_w = int(info[1])
    img_h = int(info[2])
    obj_num = int(info[3])
    bboxes = []
    labels = []
    
    for obj_idx in range(0, obj_num):
        x1 = int(float(info[4 + obj_idx * 5]))
        y1 = int(float(info[5 + obj_idx * 5]))
        x2 = int(float(info[6 + obj_idx * 5]))
        y2 = int(float(info[7 + obj_idx * 5]))
        label = int(float(info[8 + obj_idx * 5]))
        if x2 <= x1 or y2 <= y1:
            continue
        bbox = [x1, y1, x2, y2, label]
        
        labels.append(label - 1)
        bboxes.append(bbox)
    out = {
        'img_name': img_name,
        'bboxes': bboxes,
        'img_w': img_w,
        'img_h': img_h
    }
    return out

def load_annos(anno_path):
    annos = []
    with open(anno_path, "r") as f:
        for line in tqdm(f.readlines()):
            anno = annostr2dict(line)
            annos.append(anno)
    return annos


def main():
    config = '~/projects/mmclassification/configs/lw/shoes/resnet_16_16_137m_shoes.py'
    checkpoint = '/home/liwang/work_dirs/mmcls/shoes/resnet_16_16_137m_shoes/epoch_300.pth'
    device = 'cuda:3'

    save_root = '/home/liwang/data/shopee_shoes/spu_comment/images_v2_2cls_res'
    only_shoes_save_dir = os.path.join(save_root, 'only_shoes')
    shoes_with_leg_save_dir = os.path.join(save_root, 'shoes_with_leg')
    os.makedirs(only_shoes_save_dir, exist_ok=True)
    os.makedirs(shoes_with_leg_save_dir, exist_ok=True)

    # build the model from a config file and a checkpoint file
    model = init_model(config, checkpoint, device=device)



    annos =load_annos('/home/liwang/data/foot_det/shopee_shoes/anno_v2.txt')[50000:]
    for anno in tqdm(annos):
        name = anno['img_name']
        img_path = os.path.join('/home/liwang/data/shopee_shoes/spu_comment/images_v2', name)

        # test a single image
        try:
            result = inference_model(model, img_path)
        except:
            print('wrong data')
            continue
        print(result)
        if result['pred_label'] == 0:
            cmd = 'cp {} {}/'.format(img_path, only_shoes_save_dir)
        else:
            cmd = 'cp {} {}/'.format(img_path, shoes_with_leg_save_dir)

        os.system(cmd)


if __name__ == '__main__':
    main()