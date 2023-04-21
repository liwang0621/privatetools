import os
import cv2
import json
from utils import draw_lmks

pose2d_result_path = '/home/liwang/projects/Pose2Mesh_RELEASE/demo/demo_input_2dpose.json'
with open(pose2d_result_path) as f:
            pose2d_result = json.load(f)

img_name = '100271.jpg'
img_path = os.path.join('/home/liwang/projects/Pose2Mesh_RELEASE/demo/demo_input_img', img_name)
image = cv2.imread(img_path)

coco_joint_list = pose2d_result[img_name]


# for lmks in coco_joint_list:
#     image = draw_lmks(image, lmks)
image = draw_lmks(image, coco_joint_list[5])

cv2.imwrite('demo.png', image)
