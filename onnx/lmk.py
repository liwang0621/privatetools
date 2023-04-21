import cv2
import numpy as np
import onnxruntime as ort
from mmutracker.utils import box2cs, expand_bbox, keypoints_from_regression
from mmutracker.utils.image import convertLmk2Box, get_affine_transform
from mmpose.core.visualization.image import imshow_keypoints


def onnx_inference(img, model_path, config):
    model = ort.InferenceSession(model_path)
    img_ori = img.copy()
    img, c, s = preprocess(img, None, config["preprocess_cfg"])
    input_name = model.get_inputs()[0].name
    # img_ = np.load("/Users/jingshuang.liu/Documents/data/demos/img.npy")
    result = model.run(None, {input_name: img})
    result = postprocess(result[0], c, s, config["postprocess_cfg"])
    # from IPython import embed; embed()
    render_img = imshow_keypoints(img_ori, 
                    [result], 
                    skeleton=config["skeleton"],
                    kpt_score_thr=config["kpt_score_thr"],
                    pose_kpt_color=config["pose_kpt_color"],
                    pose_link_color=config["pose_link_color"])
    return result, render_img

def preprocess(img, crop_box, config):
    # np.save("/Users/jingshuang.liu/Documents/data/demos/img_onnx_0.npy", img)
    if crop_box is None:
        crop_box = [0, 0, img.shape[1]+1, img.shape[0]+1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    c, s = box2cs(crop_box, config["data_cfg"]["image_size"][0])
    print('onnx box, c,s {} {} {}'.format(crop_box, c,s))
    trans = get_affine_transform(c, s, 0, config["data_cfg"]["image_size"])
    
    img = cv2.warpAffine(
        img,
        trans, config["data_cfg"]["image_size"],
        flags=cv2.INTER_LINEAR)
    # np.save("/Users/jingshuang.liu/Documents/data/demos/img_onnx_1.npy", img)
    # from IPython import embed; embed()
    img = np.transpose(img, [2, 0, 1])
    img = img.astype(np.float32) / 255
    mean = np.asarray(config["img_norm_cfg"]["mean"], dtype=np.float32).reshape(-1, 1,1)
    std = np.asarray(config["img_norm_cfg"]["std"], dtype=np.float32).reshape(-1, 1,1)
    img = (img - mean) / std
    # np.save("/Users/jingshuang.liu/Documents/data/demos/img_onnx_2.npy", img)
    # img = np.transpose(img, [2, 0, 1])
    img = img[np.newaxis, :, :, :]

    return img, c, s


def postprocess(result, center, scale, config):
    # prepare data
    img_meta = {
        'center':
        center,
        'scale':
        scale    
    }
    img_metas = [img_meta]
    batch_size = len(img_metas)

    c = np.zeros((batch_size, 2), dtype=np.float32)
    s = np.zeros((batch_size, 2), dtype=np.float32)
    for i in range(batch_size):
        c[i, :] = img_metas[i]['center']
        s[i, :] = img_metas[i]['scale']

    preds, maxvals = keypoints_from_regression(
    result[..., :2],
    c,
    s,
    config["image_size"])
    all_preds = np.zeros((preds.shape[1], 3), dtype=np.float32)
    all_preds[:, 0:2] = preds[:, :, 0:2]
    all_preds[:, 2:3] = maxvals
    
    return all_preds


def unittest():
    model_path = "/Users/jingshuang.liu/Documents/data/demos/mobilenetv2_0.5_coco_onehand10k_swin_yydata_wlasl_fingerdance_freihand_256x256_rlefocal_1_8.onnx"
    model = ort.InferenceSession(model_path)
    input_name = model.get_inputs()[0].name
    img = np.load("/Users/jingshuang.liu/Documents/data/demos/img.npy")
    result = model.run(None, {input_name: img})
    result_pth = np.load("/Users/jingshuang.liu/Documents/data/demos/output.npy")
    assert result[0] == result_pth

if __name__ == '__main__':
    unittest()
