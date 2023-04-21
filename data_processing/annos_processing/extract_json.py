'''
Author       : Li Wang
Date         : 2022-04-26 14:46:48
LastEditors  : Li Wang
LastEditTime : 2022-04-26 15:07:33
FilePath     : /privatetools/data_processing/annos_processing/extract_json.py
Description  : 
'''

import json
import os
from tqdm import tqdm


def extract_jsons(input_dir, padding=0, code='人脸'):
    box_num, mask_num = 0, 0
    annos = []
    for json_name in tqdm(os.listdir(input_dir)):
        if '.json' not in json_name:
            continue
        with open(os.path.join(input_dir, json_name), 'r') as jsonfile:
            image = json.load(jsonfile)['result']
            width, height = image['resourceinfo']['width'] - padding * 2, image['resourceinfo']['height'] - padding * 2
            boxes = list(filter(lambda box: code in box['code'], image['data']))
            anno = json_name.replace(".json", ".png")
            anno += " %d %d %d" % (width, height, len(boxes))
            for box in boxes:
                coords = box['coordinate']
                assert len(coords) > 0, "no valid coords"
                x1, y1 = 10000, 10000
                x2, y2 = -10000, -10000
                for coord in coords:
                    x1 = min(x1, coord['x']) 
                    y1 = min(y1, coord['y'])
                    x2 = max(x2, coord['x'])
                    y2 = max(y2, coord['y'])
                
                if is_mask(box):
                    label = 0
                    mask_num += 1
                else:
                    label = 1
                    box_num += 1
                anno += " %f %f %f %f %d" % (x1 - padding, y1 - padding, x2 - padding, y2 - padding, label)
            annos.append(anno)
    print(box_num, mask_num)
    return annos     


def is_mask(box):
    for attr in box['labelAttrs']:
        if attr['title'] == '蒙版' and ('蒙版' in attr['value'] or '大蒙版' in attr['value']):
            return True
    return False


def main():
    annos1 = extract_jsons("//Users/wang.li/Downloads/【人脸人手】总数据汇总/第二批人脸人手（1）标注后数据")
    annos1_padded = extract_jsons("/Users/wang.li/Downloads/【人脸人手】总数据汇总/第二批人脸人手数据（1）挂起_padded.zip/第二批人脸人手数据（1）挂起_padded", 500)
    annos2 = extract_jsons("/Users/wang.li/Downloads/【人脸人手】总数据汇总/第二批人脸人手（2）标注后数据")
    annos2_padded = extract_jsons("/Users/wang.li/Downloads/【人脸人手】总数据汇总/第二批人脸人手数据（2）挂起_padded.zip/第二批人脸人手数据（2）挂起_padded", 500)
    annos3 = extract_jsons("/Users/wang.li/Downloads/【人脸人手】总数据汇总/第二批人脸人手数据（3）.zip")
    annos3_padded = extract_jsons("/Users/wang.li/Downloads/【人脸人手】总数据汇总/第二批人脸人手数据（3）挂起_padded.zip/第二批人脸人手数据（3）挂起_padded", 500)
    annos4 = extract_jsons("/Users/wang.li/Downloads/【人脸人手】总数据汇总/第二批人脸人手数据（4）.zip")
    annos4_padded = extract_jsons("/Users/wang.li/Downloads/【人脸人手】总数据汇总/第二批人脸人手数据（4）挂起_padded.zip/第二批人脸人手数据（4）挂起_padded", 500)
    annos5 = extract_jsons("/Users/wang.li/Downloads/【人脸人手】总数据汇总/第二批人脸人手数据（5）.zip")
    annos5_padded = extract_jsons("/Users/wang.li/Downloads/【人脸人手】总数据汇总/第二批人脸人手数据（5）挂起_padded.zip/第二批人脸人手数据（5）挂起_padded", 500)
    annos6 = extract_jsons("/Users/wang.li/Downloads/【人脸人手】总数据汇总/第二批人脸人手数据（6）.zip")
    annos6_padded = extract_jsons("/Users/wang.li/Downloads/【人脸人手】总数据汇总/第二批人脸人手数据（6）挂起_padded.zip/第二批人脸人手数据（6）挂起_padded", 500)
    annos7 = extract_jsons("/Users/wang.li/Downloads/【人脸人手】总数据汇总/第二批人脸人手数据（7）.zip")
    annos7_padded = extract_jsons("/Users/wang.li/Downloads/【人脸人手】总数据汇总/第二批人脸人手数据（7）挂起_padded.zip/第二批人脸人手数据（7）挂起_padded", 500)
    
    
    annos = annos1 + annos1_padded + annos2 + annos2_padded + annos3 + annos3_padded + annos4 + annos4_padded + annos5 + annos5_padded + annos6 + annos6_padded + annos7 + annos7_padded
    with open(os.path.join("/Users/wang.li/Downloads/anno_train_labeled.txt"), "w") as hs:
        for anno in tqdm(annos):
            hs.write(anno + "\n")


if __name__ == '__main__':
    main()
