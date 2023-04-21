'''
Author       : Li Wang
Date         : 2022-04-11 16:54:58
LastEditors  : Li Wang
LastEditTime : 2022-04-11 18:38:51
FilePath     : /privatetools/data_processing/pseudo_annotation/pose_estimation.py
Description  : 
'''
import os
import cv2
import sys
import warnings

sys.path.append('/home/liwang/projects/mmdetection')
from mmdet.apis import init_detector, inference_detector

sys.path.append('/home/liwang/projects/mmpose')
from mmpose.apis import init_pose_model, inference_top_down_pose_model, process_mmdet_results, vis_pose_result
from mmpose.datasets import DatasetInfo

class PoseEstimation:
    def __init__(
        self,
        config_path,
        checkpoint_path,
        det_config_path='/home/liwang/projects/mmdetection/configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py',
        det_checkpoint_path='https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth',
        det_persion_id=1,
        bbox_thr=0.3,
        kpt_thr=0.3,
        radius=4,
        thickness=1,
        debug=False,
        device="cuda"
    ):
        self.debug=debug
        # init mmdet
        self.det_model = init_detector(det_config_path, det_checkpoint_path, device=device)
        self.det_persion_id = det_persion_id
        self.bbox_thr = bbox_thr

        # init mmpose
        self.pose_model = init_pose_model(config_path, checkpoint_path, device=device)
        self.dataset = self.pose_model.cfg.data['test']['type']
        self.dataset_info = self.pose_model.cfg.data['test'].get('dataset_info', None)
        if self.dataset_info is None:
            warnings.warn(
                'Please set `dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                DeprecationWarning)
        else:
            self.dataset_info = DatasetInfo(self.dataset_info)
        
        self.kpt_thr = kpt_thr
        self.radius = radius
        self.thickness = thickness
        self.show = False

    def predict(self, img):
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(self.det_model, img)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, self.det_persion_id)

        # test a single image, with a list of bboxes.
        # optional
        return_heatmap = False
        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None

        pose_results, returned_outputs = inference_top_down_pose_model(
            self.pose_model,
            img,
            person_results,
            bbox_thr=self.bbox_thr,
            format='xyxy',
            dataset=self.dataset,
            dataset_info=self.dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        # show the results
        if self.debug:
            out_file = 'demo.png'
            vis_pose_result(
                self.pose_model,
                img,
                pose_results,
                dataset=self.dataset,
                dataset_info=self.dataset_info,
                kpt_score_thr=self.kpt_thr,
                radius=self.radius,
                thickness=self.thickness,
                show=self.show,
                out_file=out_file)

        return pose_results



# if __name__=='__main__':
#     config_path = '/home/liwang/projects/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vipnas_res50_coco_256x192.py'
#     checkpoint_path = 'https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_res50_coco_256x192-cc43b466_20210624.pth'
#     # 0: nose
#     # 1: left_eye
#     # 2: right_eye
#     # 3: left_ear
#     # 4: right_ear
#     # 5: left_shoulder
#     # 6: right_shoulder
#     # 7: left_elbow
#     # 8: right_elbow
#     # 9: left_wrist
#     # 10: right_wrist
#     # 11: left_hip
#     # 12: right_hip
#     # 13: left_knee
#     # 14: right_knee
#     # 15: left_ankle
#     # 16: right_ankle

#     # config_path = '/home/liwang/projects/mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/vipnas_res50_coco_wholebody_256x192_dark.py'
#     # checkpoint_path = 'https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_res50_wholebody_256x192_dark-67c0ce35_20211112.pth'


#     pose_estimation = PoseEstimation(config_path, checkpoint_path)

#     img = cv2.imread('input.png')
#     pose_results = pose_estimation.predict(img)



    