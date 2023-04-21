import cv2
import numpy as np
import onnxruntime as ort
from IPython import embed


def resize(img, resize_cfg):

    h, w, _ = img.shape
    max_long_edge = resize_cfg["longer_size"]
    max_short_edge = resize_cfg["shorter_size"]
    scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))

    new_size = int(w * float(scale_factor) + 0.5), int(h * float(scale_factor) + 0.5)

    resized_img = cv2.resize(img, new_size, dst=None, interpolation=cv2.INTER_LINEAR)
    w_scale = new_size[0] / w
    h_scale = new_size[1] / h
    return resized_img, (w_scale, h_scale)


def normalize(img, mean, std, to_rgb=True):
    # embed()
    # assert img.dtype != np.uint8
    img = np.float32(img)
    mean = np.float32(mean)
    stdinv = 1 / np.float32(std)
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    img = (img - mean) * stdinv
    return img

def pad(img, size_divisor=32, pad_val=0):
    pad_h = int(np.ceil(img.shape[0] / size_divisor)) * size_divisor
    pad_w = int(np.ceil(img.shape[1] / size_divisor)) * size_divisor

    padding = (0, 0, pad_w - img.shape[1], pad_h - img.shape[0])
    img = cv2.copyMakeBorder(
        img,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        cv2.BORDER_CONSTANT,
        value=pad_val)
    return img

def prepocess(img, preprocess_cfg):
    resize_img, resize_scale = resize(img, preprocess_cfg["resize_cfg"])
    mean = np.asarray(preprocess_cfg["img_norm_cfg"]["mean"], dtype=np.float32)
    std = np.asarray(preprocess_cfg["img_norm_cfg"]["std"], dtype=np.float32)
    resize_img = normalize(resize_img, mean, std)
    resize_img = pad(resize_img, preprocess_cfg["resize_cfg"]["size_divisor"])
    img_data = np.transpose(resize_img, [2, 0, 1])
    return img_data, resize_scale


def generate_anchors(octave_base_scale, scales_per_octave, ratios, map_h, map_w, stride):
    def _gen_base_anchors(base_size, scales, ratios):
        w = base_size
        h = base_size
        x_center = 0
        y_center = 0
        h_ratios = np.sqrt(ratios)
        w_ratios = 1 / h_ratios
        ws = (w * w_ratios[:, None] * scales[None, :])
        hs = (h * h_ratios[:, None] * scales[None, :])

        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors = np.concatenate(base_anchors, axis=-1)
        return base_anchors

    def _meshgrid(x, y, stride):
        xx = np.tile([idx for idx in range(0, x)], y).reshape((-1, 1)) * stride
        yy = np.transpose(np.tile([idx for idx in range(0, y)], x).reshape((x, -1))).reshape((-1, 1)) * stride
        return yy, xx

    octave_scales = np.array([2 ** (i / scales_per_octave) for i in range(scales_per_octave)]) * octave_base_scale
    base_anchors = _gen_base_anchors(stride, octave_scales, ratios)
    yy, xx = _meshgrid(map_w, map_h, stride)
    shifts = np.concatenate([xx, yy, xx, yy], axis=-1)
    total_anchors = base_anchors[None, :, :] + shifts[:, None, :]
    total_anchors = total_anchors.reshape((-1, 4)).astype(np.float32)
    return total_anchors


def nms(init_bboxes, iou_thres):
    def _get_iou(bbox_a, bbox_b):
        x1 = np.maximum(bbox_a[:, 0], bbox_b[0])
        y1 = np.maximum(bbox_a[:, 1], bbox_b[1])
        x2 = np.minimum(bbox_a[:, 2], bbox_b[2])
        y2 = np.minimum(bbox_a[:, 3], bbox_b[3])
        overlap = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        area_a = (bbox_a[:, 2] - bbox_a[:, 0]) * (bbox_a[:, 3] - bbox_a[:, 1])
        area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
        return overlap / (area_a + area_b - overlap)

    valid_bboxes = []
    while init_bboxes.shape[0]:
        curr_bbox = init_bboxes[0]
        ious = _get_iou(init_bboxes[1:], curr_bbox)
        valid_idxes = np.where(ious < iou_thres)[0]
        init_bboxes = init_bboxes[valid_idxes + 1]
        valid_bboxes.append(curr_bbox)
    return np.asarray(valid_bboxes).reshape((-1, init_bboxes.shape[1]))


def postprocess(raw_outputs, resize_scale, ori_shape, postprocess_cfg):
    num_classes = postprocess_cfg["num_classes"]

    anchor_cfg = postprocess_cfg["anchor_cfg"]
    strides = anchor_cfg["strides"]
    stride_num = len(strides)
    assert stride_num * 2 == len(raw_outputs)

    # generate anchors
    level_anchors = []
    for level_idx in range(0, stride_num):
        map_h = raw_outputs[level_idx].shape[2]
        map_w = raw_outputs[level_idx].shape[3]
        octave_base_scale = anchor_cfg["octave_base_scale"]
        scales_per_octave = anchor_cfg["scales_per_octave"]
        ratios = anchor_cfg["ratios"]
        stride = strides[level_idx]
        anchors = generate_anchors(octave_base_scale, scales_per_octave, ratios, map_h, map_w, stride)
        level_anchors.append(anchors)

    # get score threshold
    min_score = postprocess_cfg["min_score"]
    score_thres = np.log(min_score / (1 - min_score))

    # extract bboxes level by level
    det_bboxes = []
    for class_idx in range(0, num_classes):
        # get bboxes among levels
        init_bboxes = []
        for level_idx in range(0, stride_num):
            score_output = raw_outputs[level_idx][0]
            offset_output = raw_outputs[stride_num + level_idx][0]
            class_scores = score_output[class_idx].reshape((-1, 1))
            class_offsets = np.transpose(offset_output[4 * class_idx:4 * (class_idx + 1)].reshape((4, -1)))

            # filter score & decode bbox
            valid_idxes = np.where(class_scores >= score_thres)[0]
            valid_scores = class_scores[valid_idxes].reshape((-1, 1))
            valid_offsets = class_offsets[valid_idxes].reshape((-1, 4)) * strides[level_idx]
            valid_offsets[:, 0] *= -1
            valid_offsets[:, 1] *= -1
            valid_anchors = level_anchors[level_idx][valid_idxes].reshape((-1, 4))
            ctx = (valid_anchors[:, 0] + valid_anchors[:, 2]) * 0.5
            cty = (valid_anchors[:, 1] + valid_anchors[:, 3]) * 0.5
            valid_bboxes = valid_offsets
            valid_bboxes[:, 0::2] += ctx.reshape((-1, 1))
            valid_bboxes[:, 1::2] += cty.reshape((-1, 1))
            valid_results = np.concatenate((valid_bboxes, valid_scores), axis=-1)
            init_bboxes.append(valid_results)
        init_bboxes = np.concatenate(init_bboxes, axis=0)

        # sort bboxes by score
        sorted_idxes = np.argsort(-init_bboxes[:, 4])
        num_pre = postprocess_cfg.get("num_pre", -1)
        if num_pre > 0 and sorted_idxes.shape[0] > num_pre:
            sorted_idxes = sorted_idxes[:num_pre]
        init_bboxes = init_bboxes[sorted_idxes]

        # nms
        iou_thres = postprocess_cfg.get("iou_threshold", 0.5)
        nms_bboxes = nms(init_bboxes, iou_thres)
        nms_bboxes[:, 0] /= resize_scale[0]
        nms_bboxes[:, 1] /= resize_scale[1]
        nms_bboxes[:, 2] /= resize_scale[0]
        nms_bboxes[:, 3] /= resize_scale[1]
        nms_bboxes[:, 0] = np.minimum(np.maximum(nms_bboxes[:, 0], 0), ori_shape[1] - 1)
        nms_bboxes[:, 1] = np.minimum(np.maximum(nms_bboxes[:, 1], 0), ori_shape[0] - 1)
        nms_bboxes[:, 2] = np.minimum(np.maximum(nms_bboxes[:, 2], 0), ori_shape[1] - 1)
        nms_bboxes[:, 3] = np.minimum(np.maximum(nms_bboxes[:, 3], 0), ori_shape[0] - 1)
        # recover sigmoid score
        nms_bboxes[:, 4] = 1.0 / (1 + np.exp(-nms_bboxes[:, 4]))
        det_bboxes.append(nms_bboxes)
    return det_bboxes


def onnx_inference(img, onnx_model, model_cfg):
    # prepocess
    img_data, resize_scale = prepocess(img, model_cfg["preprocess_cfg"])
    print(img_data.shape, resize_scale)

    # forward model
    raw_outputs = onnx_model.run(None, {"input": img_data[np.newaxis, :, :, :]})

    # postprocess
    ori_shape = (img.shape[0], img.shape[1])
    results = postprocess(raw_outputs, resize_scale, ori_shape, model_cfg["postprocess_cfg"])
    return results


def render(img, results):
    classes = ['Hand']
    for label_idx in range(0, len(results)):
        b = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        r = np.random.randint(0, 256)
        color = (b, g, r)
        bboxes = results[label_idx]
        for bbox_idx in range(0, bboxes.shape[0]):
            x1 = int(bboxes[bbox_idx, 0])
            y1 = int(bboxes[bbox_idx, 1])
            x2 = int(bboxes[bbox_idx, 2])
            y2 = int(bboxes[bbox_idx, 3])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        for bbox_idx in range(0, bboxes.shape[0]):
            x1 = int(bboxes[bbox_idx, 0])
            y1 = int(bboxes[bbox_idx, 1])
            score = bboxes[bbox_idx, 4]
            class_name = classes[label_idx]
            y_text = y1 + 20
            render_text = "{}:{:3.2f}".format(class_name, score)
            cv2.putText(img, render_text, (x1, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    return img


if __name__ == "__main__":
    #onnx_model_path = "mobilenet_35_rgb_shared_stacked_convs.onnx"
    onnx_model_path = "/Users/wang.li/remote_projects/mmdetection/output/gfl_ghostnet_foot_coco_sjt_labeled_v1v2v3x2_480.onnx"
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
            nms_thres=0.5
        )
    )

    img_path = "/Users/wang.li/data/demo_images/foot_demos/1.jpg"
    img = cv2.imread(img_path)

    onnx_model = ort.InferenceSession(onnx_model_path)
    results = onnx_inference(img, onnx_model, model_cfg)
    print(results)
    # render_img = render(img, results)
    # cv2.imwrite("/Users/jingshuang.liu/Documents/data/demos/onnx_demos/render_img1.jpg", render_img)
