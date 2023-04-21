import os
import cv2
from glob import glob
from tqdm import tqdm

img_num = len(glob('/home/liwang/projects/Pose2Mesh_RELEASE/demo/result/*mesh.png'))

height, width, _ = cv2.imread('/home/liwang/projects/Pose2Mesh_RELEASE/demo/result/{}_mesh.png'.format(0)).shape


#写入地址
path = "human_mesh_demo1.avi"
#设置宽高
size = (int(width),int(height))
#设置帧率
fps = 49
#cv2.VideoWriter_fourcc('I','4','2','0')设置编码格式
# video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)
video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)


for i in tqdm(range(img_num)):
    img_path = '/home/liwang/projects/Pose2Mesh_RELEASE/demo/result/{}_mesh.png'.format(i)
    # print(img_path)
    # print(os.path.exists(img_path))
    img = cv2.imread(img_path)
    # 图片写入视频
    video.write(img)

video.release()

