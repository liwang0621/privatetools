import os
import cv2
import sys
import numpy as np
from glob import glob
from tqdm import tqdm
import mediapipe as mp
import multiprocessing
from multiprocessing import Manager
from loguru import logger
sys.path.append('../')
from annos_processing.utils import load_annos, save_annos
mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

def create_bbox_by_keypoints(keypoints, label=1, enlarge_ratio=1.35):
    min_short_long_ratio = 0.35

    if keypoints.shape[0] == 0:
        return None
    try:
        ori_x1 = np.min(keypoints[:, 0])
    except:
        from pdb import set_trace;set_trace()

    ori_y1 = np.min(keypoints[:, 1])
    ori_x2 = np.max(keypoints[:, 0])
    ori_y2 = np.max(keypoints[:, 1])
    ctx = (ori_x1 + ori_x2) * 0.5
    cty = (ori_y1 + ori_y2) * 0.5
    ori_w = abs(ori_x2 - ori_x1)
    ori_h = abs(ori_y2 - ori_y1)
    min_size = max(ori_w, ori_h) * min_short_long_ratio
    ori_w = max(ori_w, min_size)
    ori_h = max(ori_h, min_size)

    enlarge_w = ori_w * enlarge_ratio
    enlarge_h = ori_h * enlarge_ratio
    x1 = ctx - enlarge_w * 0.5
    y1 = cty - enlarge_h * 0.5
    x2 = ctx + enlarge_w * 0.5
    y2 = cty + enlarge_h * 0.5
    return [x1, y1, x2, y2]
def draw_box(img, boxes):
    b = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    r = np.random.randint(0, 256)
    color = (b, g, r)
    for box in boxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img

def worker(procnum, image_paths_worker, return_dict):
    annos = []
    pabr = tqdm(total=len(image_paths_worker))
    with mp_objectron.Objectron(static_image_mode=True,
                            max_num_objects=5,
                            min_detection_confidence=0.7,
                            model_name='Shoe') as objectron:
        for idx, file in enumerate(image_paths_worker):
            image = cv2.imread(file)
            name = os.path.basename(file)
            # Convert the BGR image to RGB and process it with MediaPipe Objectron.
            try:
                results = objectron.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            except:
                print('wrong {}'.format(file))
            pabr.update(1)

            # Draw box landmarks.
            if not results.detected_objects:
                print(f'No box landmarks detected on {file}')
                continue
            print(f'Box landmarks of {file}:')
            annotated_image = image.copy()
            h, w, _ = annotated_image.shape
            boxes = []
            for detected_object in results.detected_objects:
                # from pdb import set_trace; set_trace()
                # mp_drawing.draw_landmarks(
                #     annotated_image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                # mp_drawing.draw_axis(annotated_image, detected_object.rotation,
                #                     detected_object.translation)
                lmks = []
                for idx, landmark in enumerate(detected_object.landmarks_2d.landmark):
                    lmks.append([landmark.x * w, landmark.y * h])
                box = create_bbox_by_keypoints(np.array(lmks), label=1, enlarge_ratio=1)
                box.append(1)
                boxes.append(box)
            anno = {
                'img_name': name,
                'bboxes': boxes,
                'img_w': w,
                'img_h': h
            }
            annos.append(anno)
    return_dict[procnum] = annos

# For static images:
# IMAGE_FILES = [
#     '871655105330_.pic.jpg',
#     '15071655102318_.pic.jpg'
# ]
image_list = glob('/home/liwang/data/shopee_shoes/spu_comment/images_v2/*.jpg')


multithread = 8

nr_lines = len(image_list)
nr_liner_per_worker = np.ceil(nr_lines / multithread)


manager = Manager()
# return_list = manager.list() 也可以使用列表list
return_dict = manager.dict()
jobs = []
for i in range(multithread):
    device = 'cuda:{}'.format(i%8)

    start = int(i * nr_liner_per_worker)                                               
    end = int(min(start + nr_liner_per_worker, nr_lines))
    image_paths_worker = image_list[start:end]

    p = multiprocessing.Process(target=worker, args=(i, image_paths_worker, return_dict))
    jobs.append(p)
    p.start()

for proc in jobs:
    proc.join()
    
result_dict = dict(return_dict)

annos = []
for key in result_dict:
    annos.extend(result_dict[key])

save_anno_path = '/home/liwang/data/foot_det/shopee_shoes/anno_v2.txt'
save_annos(annos, save_anno_path)