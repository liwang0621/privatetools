import os
from glob import glob
from tqdm import tqdm
import sys
sys.path.append('~/projects/mmclassification/')
from mmcls.apis import inference_model, init_model, show_result_pyplot


def main():
    config = '~/projects/mmclassification/configs/lw/shoes/resnet_16_16_137m_shoes.py'
    checkpoint = '/home/liwang/work_dirs/mmcls/shoes/resnet_16_16_137m_shoes_v2/epoch_300.pth'
    device = 'cuda:5'
    # img_path = '/home/liwang/data/shopee_shoes/spu_comment/images/000377229c19734b5071f30f7aacf428.jpg'

    save_root = '/home/liwang/data/shopee_shoes/spu_comment/video_10w_cls_res'
    # only_shoes_save_dir = os.path.join(save_root, 'only_shoes')
    shoes_with_leg_save_dir = os.path.join(save_root, 'shoes_with_leg')
    # os.makedirs(only_shoes_save_dir, exist_ok=True)
    os.makedirs(shoes_with_leg_save_dir, exist_ok=True)

    # build the model from a config file and a checkpoint file
    model = init_model(config, checkpoint, device=device)

    for img_path in tqdm(glob('/home/liwang/data/shopee_shoes/spu_comment/videos_v2_2_img_10w/*.jpg')[500000:]):
        # test a single image
        try:
            result = inference_model(model, img_path)
        except:
            # from IPython import embed; embed()
            # result = inference_model(model, img_path)
            print('wrong data')
            continue
        print(result)
        # if result['pred_label'] == 0:
        #     cmd = 'cp {} {}/'.format(img_path, only_shoes_save_dir)
        # else:
        #     cmd = 'cp {} {}/'.format(img_path, shoes_with_leg_save_dir)
        if result['pred_label'] == 1:
            cmd = 'cp {} {}/'.format(img_path, shoes_with_leg_save_dir)

            os.system(cmd)
    # # show the results
    # show_result_pyplot(model, img_path, result)


if __name__ == '__main__':
    main()