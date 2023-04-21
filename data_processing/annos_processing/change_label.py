from utils import load_annos, save_annos

anno_path = '/home/liwang/data/product_det/ShopeeProduct/anno_test.txt'
save_anno_path = '/home/liwang/data/product_det/ShopeeProduct/anno_1class_test.txt'
new_label = 1
annos = load_annos(anno_path)

for anno in annos:
    for box in anno['bboxes']:
        box[-1] = new_label
save_annos(annos, save_anno_path)