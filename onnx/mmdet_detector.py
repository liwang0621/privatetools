import os
from IPython.terminal.embed import embed
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mmdet.apis import init_detector, inference_detector
import cv2
import mmcv
import os
import numpy as np

def load_img_paths(img_dir):
    img_paths = []
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        img_paths.append(img_path)
    return img_paths

def render_result(img, bboxes, classes, colors):
    for bbox in bboxes:
        class_idx, _, x1, y1, x2, y2 = bbox
        color = colors[class_idx]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    for bbox in bboxes:
        class_idx, score, x1, y1, x2, y2 = bbox
        class_name = classes[class_idx]
        color = colors[class_idx]
        y_text = y1 + 20
        render_text = "{}:{:3.2f}".format(class_name, score)
        cv2.putText(img, render_text, (int(x1), int(y_text)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

class Detector:
    def __init__(self, config_path, checkpoint_path, min_score, device="cuda"):
        config = mmcv.Config.fromfile(config_path)
        config.model.test_cfg.score_thr = min_score
        self.detector = init_detector(config, checkpoint_path, device=device)
        self.classes = config["classes"]
        self.colors = []
        for _ in range(0, len(self.classes)):
            b = np.random.randint(0, 256)
            g = np.random.randint(0, 256)
            r = np.random.randint(0, 256)
            self.colors.append((b, g, r))

    def predict(self, img):
        result = inference_detector(self.detector, [img])[0]
        bboxes = []
        for class_idx, class_result in enumerate(result):
            for bbox_idx in range(0, class_result.shape[0]):
                x1 = min(max(int(class_result[bbox_idx][0]), 0), img.shape[1] - 1)
                y1 = min(max(int(class_result[bbox_idx][1]), 0), img.shape[0] - 1)
                x2 = min(max(int(class_result[bbox_idx][2]), 0), img.shape[1] - 1)
                y2 = min(max(int(class_result[bbox_idx][3]), 0), img.shape[0] - 1)
                score = class_result[bbox_idx][4]
                bboxes.append((class_idx, score, x1, y1, x2, y2))
        return bboxes
