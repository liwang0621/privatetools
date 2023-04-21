import onnxruntime as ort
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mmdet_detector import Detector
from onnx_inference import onnx_inference
from lmk import onnx_inference as lmk_onnx_inference
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
import numpy as np
import cv2


palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255],
                    [255, 0, 0], [255, 255, 255]])


def det_onnx_res(img_path = "/home/liwang/data/demo_images/foot_demos/1.jpg"):
    #onnx_model_path = "mobilenet_35_rgb_shared_stacked_convs.onnx"
    onnx_model_path = "/home/liwang/projects/mmdetection/output/gfl_ghostnet_foot_coco_sjt_labeled_v1v2v3x2_480.onnx"
    model_cfg = dict(
        preprocess_cfg=dict(
            resize_cfg=dict(longer_size=480, shorter_size=480, size_divisor=32),
            img_norm_cfg=dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        ),
        postprocess_cfg=dict(
            anchor_cfg=dict(
                strides=[8, 16, 32, 64],
                ratios=[1.0],
                octave_base_scale=3,
                scales_per_octave=1,
            ),
            num_classes=1,
            min_score=0.5,
            nms_pre=100,
            nms_thres=0.1
        )
    )
    img = cv2.imread(img_path)

    onnx_model = ort.InferenceSession(onnx_model_path)
    return onnx_inference(img, onnx_model, model_cfg)


def det_pth_res(img_path = "/home/liwang/data/demo_images/foot_demos/1.jpg"):
    config_path = '/home/liwang/projects/mmdetection/configs/lw/foot_det/gfl_ghostnet_foot_coco_sjt_labeled_v1v2v3x2_480.py'
    checkpoint_path = '/home/liwang/work_dirs/foot_det/gfl_ghostnet_foot_coco_sjt_labeled_v1v2v3x2_480/epoch_180.pth'
    min_score = 0.4

    detector = Detector(config_path, checkpoint_path, min_score, 'cpu')
    img = cv2.imread(img_path)
    return detector.predict(img)


def det_consistency_check():
    res1 = det_onnx_res()
    res2 = det_pth_res()
    print(res1)
    print(res2)
    assert res1 == res2


def lmk_pth_res(img_path = "/home/liwang/data/demo_images/foot_demos/crop.jpg"):
    config_path = '/home/liwang/projects/mmpose/configs/lw/foot/deeppose_x14m_sjt_v1v2v3_exp0_9p_224x224_rle.py'
    checkpoint_path = '/home/liwang/work_dirs/mmpose/foot/deeppose_x14m_sjt_v1v2v3_exp0_9p_224x224_rle/epoch_210.pth'

    pose_model = init_pose_model(
        config_path, checkpoint_path, 'cpu')
    img = cv2.imread(img_path)
    img = cv2.flip(img, 1)
    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        img,
        None,
        bbox_thr=None,
        format='xyxy',
        dataset='OneHand10KDataset',
        dataset_info=None,
        return_heatmap=False,
        outputs=None)
    print('lmk mmpose res: {}'.format(pose_results))
    vis_pose_result(
        pose_model,
        img,
        pose_results,
        dataset='OneHand10KDataset',
        dataset_info=None,
        kpt_score_thr=0,
        radius=4,
        thickness=1,
        show=False,
        out_file="/home/liwang/data/output/d.png")
    return pose_results


def lmk_onnx_res(img_path = "/home/liwang/data/demo_images/foot_demos/crop.jpg"):
    model_path = "/home/liwang/projects/mmpose/output/deeppose_x14m_sjt_v1v2v3_exp0_9p_224x224_rle.onnx"
    config = dict(
                    preprocess_cfg=dict(
                        expand_ratio=1.0,
                        data_cfg=dict(image_size=[224, 224]),
                        img_norm_cfg=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
                    ),
                    postprocess_cfg=dict(
                        post_process="default",
                        kernel=11,
                        valid_radius_factor=0.0546875,
                        use_udp=False,
                        target_type="GaussianHeatmap",
                        unbiased_decoding=False,
                        lmk_thre=0.3,
                        image_size=[224, 224]
                    ),
                    kpt_score_thr=0.1,
                    skeleton = [
                            [0,2],
                            [2,7],
                            [5,7],
                            [6,7],
                            [7,8],
                            [1,8],
                            [1,3],
                            [1,4],
                        ],
                    pose_link_color=palette[[
                        0, 0, 0, 0, 4, 4, 4, 4, 
                    ]],
                    pose_kpt_color=palette[[
                        0, 0, 0, 0, 0, 4, 4, 4, 4,
                    ]]
            )
    img = cv2.imread(img_path)
    img = cv2.flip(img, 1)
    result, image = lmk_onnx_inference(img, model_path, config)
    print('lmk onnx res: {}'.format(result))
    cv2.imwrite("/home/liwang/data/output/dd.png", image)
    return result


def lmk_consistency_check():
    res1 = lmk_pth_res()
    res2 = lmk_onnx_res()


def input_consistency_check():
    for i in range(3):
        onnx_img = np.load("/Users/jingshuang.liu/Documents/data/demos/img_onnx_{}.npy".format(i))
        pth_img = np.load("/Users/jingshuang.liu/Documents/data/demos/img_pth_{}.npy".format(i))
        assert onnx_img == pth_img


if __name__ == '__main__':
    # det_consistency_check()
    lmk_consistency_check()
    # input_consistency_check()
