from pycocotools.coco import COCO
import numpy as np
import cv2
import os

def create_bbox_by_keypoints(keypoints, label, enlarge_ratio=1.3):
    if label == 1:
        min_short_long_ratio = 0.6
    else:
        min_short_long_ratio = 0.35
    if np.sum(keypoints[:, 2] > 0) != keypoints.shape[0]:
        return None
    # keypoints = keypoints[keypoints[:, 2] > 0]
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
    return np.asarray([x1, y1, x2, y2])

def render(img_name, img_anns, img_dir):
    img_path = os.path.join(img_dir, img_name)
    r_img = cv2.imread(img_path).copy()
    for img_ann in img_anns:
        bbox, label = img_ann
        r_bbox = bbox.astype(np.int32)
        if label == -1:
            color = (0, 0, 255)
        elif label == 1:
            color = (0, 255, 0)
        else:
            color = (255, 255, 0)
        cv2.rectangle(r_img, (r_bbox[0], r_bbox[1]), (r_bbox[2], r_bbox[3]), color, 2)
    return r_img

def extract_coco_anns(ann_path, upper_label=1, lower_label=2):
    coco = COCO(ann_path)
    cat_ids = coco.getCatIds()
    img_ids = coco.getImgIds(catIds=cat_ids)
    imgs = coco.loadImgs(img_ids)
    #imgs = imgs[0:10]

    # 0: nose
    # 1: left_eye
    # 2: right_eye
    # 3: left_ear
    # 4: right_ear
    # 5: left_shoulder
    # 6: right_shoulder
    # 7: left_elbow
    # 8: right_elbow
    # 9: left_wrist
    # 10: right_wrist
    # 11: left_hip
    # 12: right_hip
    # 13: left_knee
    # 14: right_knee
    # 15: left_ankle
    # 16: right_ankle
    keypoint_names = coco.loadCats(cat_ids)[0]["keypoints"]

    img_anns = []
    for img in imgs:
        img_name = img["file_name"]
        #if img_name != "000000000113.jpg":
        #    continue
        ann_ids = coco.getAnnIds(imgIds=img["id"], catIds=cat_ids)
        anns = coco.loadAnns(ann_ids)
        cvt_anns = []
        is_valid = True
        for ann in anns:
            x1, y1, w, h = ann["bbox"]
            bbox = np.asarray([x1, y1, x1 + w, y1 + h])
            iscrowd = True if ann["iscrowd"] else False
            keypoints = np.asarray(ann["keypoints"]).reshape((-1, 3))
            # if '000000131101' in img_name:
            #     from pdb import set_trace;set_trace()
            upper_keypoints = keypoints[[5, 6, 11, 12], :]
            lower_keypoints = keypoints[[11, 12, 13, 14, 15, 16], :]
            upper_bbox = create_bbox_by_keypoints(upper_keypoints, upper_label)
            lower_bbox = create_bbox_by_keypoints(lower_keypoints, lower_label)
            if iscrowd or (upper_bbox is None or lower_bbox is None):
                cvt_anns.append((bbox, -1))
            # if iscrowd or (upper_bbox is None and lower_bbox is None):
            #     cvt_anns.append((bbox, -1))
            else:
                if upper_bbox is not None:
                    cvt_anns.append((upper_bbox, upper_label))
                if lower_bbox is not None:
                    cvt_anns.append((lower_bbox, lower_label))
        img_anns.append((img_name, cvt_anns))

        # render
        # img_dir = "/Users/wang.li/data/coco/train2017"
        # save_dir = "/Users/wang.li/tmp/render_train"
        img_dir = "/Users/wang.li/data/coco/val2017"
        save_dir = "/Users/wang.li/tmp/render_val"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, img_name)
        print(save_path)
        cv2.imwrite(save_path, render(img_name, cvt_anns, img_dir))
    return img_anns

if __name__ == "__main__":
    ann_path = "/Users/wang.li/data/coco/annotations/person_keypoints_val2017.json"
    # ann_path = "/Users/wang.li/data/coco/annotations/person_keypoints_train2017.json"

    save_dir = "ext_annos"
    # save_path = os.path.join(save_dir, "{}.txt".format(os.path.basename(ann_path).split(".")[0]))
    save_path = '/Users/wang.li/tmp/person_keypoints_val2017.txt'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    img_anns = extract_coco_anns(ann_path)
    from pdb import set_trace; set_trace()
    with open(save_path, "w") as f:
        for img_name, cvt_anns in img_anns:
            line = "{} {}".format(img_name, len(cvt_anns))
            for bbox, label in cvt_anns:
                line += " {} {} {} {} {}".format(bbox[0], bbox[1], bbox[2], bbox[3], label)
            f.writelines(line + "\n")
