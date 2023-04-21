import os
from utils import save_annos

annos = []
with open('/home/liwang/data/product_data/检测训练数据/train.txt', 'r') as f:
    for line in f.readlines():
        anno = {
                'img_name': None,
                'bboxes': None,
                'img_w': None,
                'img_h': None
            }
        tmp = line.split()
        name = '{}.jpeg'.format(os.path.basename(tmp[0]))
        img_w = int(tmp[1])
        img_h = int(tmp[2])
        num_boxes = int((len(tmp) - 3)/5)
        boxes = []

        for i in range(num_boxes):
            category_id = int(tmp[i*5 + 3])
            cx = float(tmp[i*5 + 4]) * img_w
            cy = float(tmp[i*5 + 5]) * img_h
            w = float(tmp[i*5 + 6]) * img_w
            h = float(tmp[i*5 + 7]) * img_h

            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            bbox = [x1, y1, x2, y2, category_id]
            boxes.append(bbox)
        anno['img_name'] = name
        anno['bboxes'] = boxes
        anno['img_w'] = img_w
        anno['img_h'] = img_h

        annos.append(anno)

save_annos(annos, '/home/liwang/data/product_det/dianshang_product/train.txt')



