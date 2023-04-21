import cv2
import random
import numpy as np


class PfCropAffine():
    def __init__(self,
                 out_shape=(256, 256),
                 expand_ratio_range=(0.2, 1),
                 shift_ratio=1):
        self.out_shape = out_shape
        self.expand_ratio_range = expand_ratio_range
        self.expand_ratio_scale = expand_ratio_range[1] - expand_ratio_range[0]
        self.shift_ratio = shift_ratio

    def get_affine_transform(self, box, out_shape):
        x, y, x2, y2 = box
        bboxW = x2 - x
        bboxH = y2 - y
        max_boxwh = max(bboxH, bboxW)
        bboxW = max_boxwh
        bboxH = max_boxwh
        outW, outH = out_shape

        trans = [[
            outW / bboxW, 0, outW / 2 - (outW / bboxW) * (x + bboxW / 2)
        ], [0, outH / bboxH, outH / 2 - (outH / bboxH) * (y + bboxH / 2)]]
        return np.array(trans)

    def expand_bbox(self, box, image_w, image_h, expand_ratio):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        expend_size = max(w, h) * (1 + expand_ratio)

        x1_ex = max(0, cx - expend_size / 2)
        y1_ex = max(0, cy - expend_size / 2)
        x2_ex = min(cx + expend_size / 2, image_w)
        y2_ex = min(cy + expend_size / 2, image_h)

        if random.random() < self.shift_ratio:
            shift_x = random.random() * (x1 - x1_ex)
            shift_y = random.random() * (y1 - y1_ex)
            x1_ex = max(0, x1_ex + shift_x)
            y1_ex = max(0, y1_ex + shift_y)
            x2_ex = min(image_w, x2_ex + shift_x)
            y2_ex = min(image_h, y2_ex + shift_y)

        return (int(x1_ex), int(y1_ex), int(x2_ex), int(y2_ex))

    def __call__(self, results):
        image_h, image_w = results['img'].shape[:2]

        boxes = []
        for key in ['gt_bboxes']:
            boxes.extend(results[key])
        assert len(boxes) == 1
        box = boxes[0]
        label = results['gt_labels']
        assert len(label) == 1
        label = label[0]
        # from pdb import set_trace;set_trace()

        expand_ratio = random.random(
        ) * self.expand_ratio_scale + self.expand_ratio_range[0]
        crop_box = self.expand_bbox(box, image_w, image_h, expand_ratio)

        crop_img = results['img'][int(crop_box[1]):int(crop_box[3]),
                                  int(crop_box[0]):int(crop_box[2]), :]
        crop_image_h, crop_image_w = crop_img.shape[:2]
        try:
            assert min(crop_image_h, crop_image_w) > 1
        except:
            print(image_h, image_w, crop_image_h, crop_image_w, int(crop_box[1]), int(crop_box[3]), int(crop_box[0]), int(crop_box[2]))

        # save_path = '/home/liwang/tmp/' + results['filename'].split('.')[0] + str(results['gt_bboxes']) +  'label_ {}'.format(results['gt_labels']) +'.jpg'
        # cv2.imwrite(save_path, crop_img)
        if label != 0:
            new_box = [0, 0] + [crop_image_w, crop_image_h]
        else:
            new_box = [
                box[0] - crop_box[0], box[1] - crop_box[1],
                box[2] - crop_box[0], box[3] - crop_box[1]
            ]

        results['img'] = crop_img
        results['img_shape'] = crop_img.shape
        results['gt_bboxes'] = np.array([new_box])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(out_shape={self.out_shape}, '
        repr_str += f'expand_ratio_range={self.expand_ratio_range}, '
        return repr_str

if __name__ == '__main__':
    img = cv2.imread('/Users/wang.li/data/demo_images/foot_demos/1.jpg')
    results = {}
    results['img'] = img
    results['gt_bboxes'] = [[450, 450, 900, 700]]
    results['gt_labels'] = [0]
    results['bbox_fields'] = ['gt_bboxes']

    aug = PfCropAffine()

    for i in range(10000):
        results = aug(results)
