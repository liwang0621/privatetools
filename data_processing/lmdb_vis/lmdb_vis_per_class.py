'''
Author       : Li Wang
Date         : 2022-04-28 16:32:54
LastEditors  : wang.li
LastEditTime : 2023-01-06 16:00:01
FilePath     : /privatetools/data_processing/lmdb_vis/lmdb_vis_per_class.py
Description  : 
'''

import os
import cv2
import sys
import lmdb
import argparse
from matplotlib.pyplot import cla
import numpy as np
from tqdm import tqdm

from utils import load_annotations, load_annotations_label
# sys.path.append('../')
# from pseudo_annotation.detector import Detector, render_result

from IPython import embed

class_map = {-1:'ignore',
 1: 'Person',
 2: 'Sneakers',
 3: 'Chair',
 4: 'Other Shoes',
 5: 'Hat',
 6: 'Car',
 7: 'Lamp',
 8: 'Glasses',
 9: 'Bottle',
 10: 'Desk',
 11: 'Cup',
 12: 'Street Lights',
 13: 'Cabinet/shelf',
 14: 'Handbag/Satchel',
 15: 'Bracelet',
 16: 'Plate',
 17: 'Picture/Frame',
 18: 'Helmet',
 19: 'Book',
 20: 'Gloves',
 21: 'Storage box',
 22: 'Boat',
 23: 'Leather Shoes',
 24: 'Flower',
 25: 'Bench',
 26: 'Potted Plant',
 27: 'Bowl/Basin',
 28: 'Flag',
 29: 'Pillow',
 30: 'Boots',
 31: 'Vase',
 32: 'Microphone',
 33: 'Necklace',
 34: 'Ring',
 35: 'SUV',
 36: 'Wine Glass',
 37: 'Belt',
 38: 'Moniter/TV',
 39: 'Backpack',
 40: 'Umbrella',
 41: 'Traffic Light',
 42: 'Speaker',
 43: 'Watch',
 44: 'Tie',
 45: 'Trash bin Can',
 46: 'Slippers',
 47: 'Bicycle',
 48: 'Stool',
 49: 'Barrel/bucket',
 50: 'Van',
 51: 'Couch',
 52: 'Sandals',
 53: 'Bakset',
 54: 'Drum',
 55: 'Pen/Pencil',
 56: 'Bus',
 57: 'Wild Bird',
 58: 'High Heels',
 59: 'Motorcycle',
 60: 'Guitar',
 61: 'Carpet',
 62: 'Cell Phone',
 63: 'Bread',
 64: 'Camera',
 65: 'Canned',
 66: 'Truck',
 67: 'Traffic cone',
 68: 'Cymbal',
 69: 'Lifesaver',
 70: 'Towel',
 71: 'Stuffed Toy',
 72: 'Candle',
 73: 'Sailboat',
 74: 'Laptop',
 75: 'Awning',
 76: 'Bed',
 77: 'Faucet',
 78: 'Tent',
 79: 'Horse',
 80: 'Mirror',
 81: 'Power outlet',
 82: 'Sink',
 83: 'Apple',
 84: 'Air Conditioner',
 85: 'Knife',
 86: 'Hockey Stick',
 87: 'Paddle',
 88: 'Pickup Truck',
 89: 'Fork',
 90: 'Traffic Sign',
 91: 'Ballon',
 92: 'Tripod',
 93: 'Dog',
 94: 'Spoon',
 95: 'Clock',
 96: 'Pot',
 97: 'Cow',
 98: 'Cake',
 99: 'Dinning Table',
 100: 'Sheep',
 101: 'Hanger',
 102: 'Blackboard/Whiteboard',
 103: 'Napkin',
 104: 'Other Fish',
 105: 'Orange/Tangerine',
 106: 'Toiletry',
 107: 'Keyboard',
 108: 'Tomato',
 109: 'Lantern',
 110: 'Machinery Vehicle',
 111: 'Fan',
 112: 'Green Vegetables',
 113: 'Banana',
 114: 'Baseball Glove',
 115: 'Airplane',
 116: 'Mouse',
 117: 'Train',
 118: 'Pumpkin',
 119: 'Soccer',
 120: 'Skiboard',
 121: 'Luggage',
 122: 'Nightstand',
 123: 'Tea pot',
 124: 'Telephone',
 125: 'Trolley',
 126: 'Head Phone',
 127: 'Sports Car',
 128: 'Stop Sign',
 129: 'Dessert',
 130: 'Scooter',
 131: 'Stroller',
 132: 'Crane',
 133: 'Remote',
 134: 'Refrigerator',
 135: 'Oven',
 136: 'Lemon',
 137: 'Duck',
 138: 'Baseball Bat',
 139: 'Surveillance Camera',
 140: 'Cat',
 141: 'Jug',
 142: 'Broccoli',
 143: 'Piano',
 144: 'Pizza',
 145: 'Elephant',
 146: 'Skateboard',
 147: 'Surfboard',
 148: 'Gun',
 149: 'Skating and Skiing shoes',
 150: 'Gas stove',
 151: 'Donut',
 152: 'Bow Tie',
 153: 'Carrot',
 154: 'Toilet',
 155: 'Kite',
 156: 'Strawberry',
 157: 'Other Balls',
 158: 'Shovel',
 159: 'Pepper',
 160: 'Computer Box',
 161: 'Toilet Paper',
 162: 'Cleaning Products',
 163: 'Chopsticks',
 164: 'Microwave',
 165: 'Pigeon',
 166: 'Baseball',
 167: 'Cutting/chopping Board',
 168: 'Coffee Table',
 169: 'Side Table',
 170: 'Scissors',
 171: 'Marker',
 172: 'Pie',
 173: 'Ladder',
 174: 'Snowboard',
 175: 'Cookies',
 176: 'Radiator',
 177: 'Fire Hydrant',
 178: 'Basketball',
 179: 'Zebra',
 180: 'Grape',
 181: 'Giraffe',
 182: 'Potato',
 183: 'Sausage',
 184: 'Tricycle',
 185: 'Violin',
 186: 'Egg',
 187: 'Fire Extinguisher',
 188: 'Candy',
 189: 'Fire Truck',
 190: 'Billards',
 191: 'Converter',
 192: 'Bathtub',
 193: 'Wheelchair',
 194: 'Golf Club',
 195: 'Briefcase',
 196: 'Cucumber',
 197: 'Cigar/Cigarette ',
 198: 'Paint Brush',
 199: 'Pear',
 200: 'Heavy Truck',
 201: 'Hamburger',
 202: 'Extractor',
 203: 'Extention Cord',
 204: 'Tong',
 205: 'Tennis Racket',
 206: 'Folder',
 207: 'American Football',
 208: 'earphone',
 209: 'Mask',
 210: 'Kettle',
 211: 'Tennis',
 212: 'Ship',
 213: 'Swing',
 214: 'Coffee Machine',
 215: 'Slide',
 216: 'Carriage',
 217: 'Onion',
 218: 'Green beans',
 219: 'Projector',
 220: 'Frisbee',
 221: 'Washing Machine/Drying Machine',
 222: 'Chicken',
 223: 'Printer',
 224: 'Watermelon',
 225: 'Saxophone',
 226: 'Tissue',
 227: 'Toothbrush',
 228: 'Ice cream',
 229: 'Hotair ballon',
 230: 'Cello',
 231: 'French Fries',
 232: 'Scale',
 233: 'Trophy',
 234: 'Cabbage',
 235: 'Hot dog',
 236: 'Blender',
 237: 'Peach',
 238: 'Rice',
 239: 'Wallet/Purse',
 240: 'Volleyball',
 241: 'Deer',
 242: 'Goose',
 243: 'Tape',
 244: 'Tablet',
 245: 'Cosmetics',
 246: 'Trumpet',
 247: 'Pineapple',
 248: 'Golf Ball',
 249: 'Ambulance',
 250: 'Parking meter',
 251: 'Mango',
 252: 'Key',
 253: 'Hurdle',
 254: 'Fishing Rod',
 255: 'Medal',
 256: 'Flute',
 257: 'Brush',
 258: 'Penguin',
 259: 'Megaphone',
 260: 'Corn',
 261: 'Lettuce',
 262: 'Garlic',
 263: 'Swan',
 264: 'Helicopter',
 265: 'Green Onion',
 266: 'Sandwich',
 267: 'Nuts',
 268: 'Speed Limit Sign',
 269: 'Induction Cooker',
 270: 'Broom',
 271: 'Trombone',
 272: 'Plum',
 273: 'Rickshaw',
 274: 'Goldfish',
 275: 'Kiwi fruit',
 276: 'Router/modem',
 277: 'Poker Card',
 278: 'Toaster',
 279: 'Shrimp',
 280: 'Sushi',
 281: 'Cheese',
 282: 'Notepaper',
 283: 'Cherry',
 284: 'Pliers',
 285: 'CD',
 286: 'Pasta',
 287: 'Hammer',
 288: 'Cue',
 289: 'Avocado',
 290: 'Hamimelon',
 291: 'Flask',
 292: 'Mushroon',
 293: 'Screwdriver',
 294: 'Soap',
 295: 'Recorder',
 296: 'Bear',
 297: 'Eggplant',
 298: 'Board Eraser',
 299: 'Coconut',
 300: 'Tape Measur/ Ruler',
 301: 'Pig',
 302: 'Showerhead',
 303: 'Globe',
 304: 'Chips',
 305: 'Steak',
 306: 'Crosswalk Sign',
 307: 'Stapler',
 308: 'Campel',
 309: 'Formula 1',
 310: 'Pomegranate',
 311: 'Dishwasher',
 312: 'Crab',
 313: 'Hoverboard',
 314: 'Meat ball',
 315: 'Rice Cooker',
 316: 'Tuba',
 317: 'Calculator',
 318: 'Papaya',
 319: 'Antelope',
 320: 'Parrot',
 321: 'Seal',
 322: 'Buttefly',
 323: 'Dumbbell',
 324: 'Donkey',
 325: 'Lion',
 326: 'Urinal',
 327: 'Dolphin',
 328: 'Electric Drill',
 329: 'Hair Dryer',
 330: 'Egg tart',
 331: 'Jellyfish',
 332: 'Treadmill',
 333: 'Lighter',
 334: 'Grapefruit',
 335: 'Game board',
 336: 'Mop',
 337: 'Radish',
 338: 'Baozi',
 339: 'Target',
 340: 'French',
 341: 'Spring Rolls',
 342: 'Monkey',
 343: 'Rabbit',
 344: 'Pencil Case',
 345: 'Yak',
 346: 'Red Cabbage',
 347: 'Binoculars',
 348: 'Asparagus',
 349: 'Barbell',
 350: 'Scallop',
 351: 'Noddles',
 352: 'Comb',
 353: 'Dumpling',
 354: 'Oyster',
 355: 'Table Teniis paddle',
 356: 'Cosmetics Brush/Eyeliner Pencil',
 357: 'Chainsaw',
 358: 'Eraser',
 359: 'Lobster',
 360: 'Durian',
 361: 'Okra',
 362: 'Lipstick',
 363: 'Cosmetics Mirror',
 364: 'Curling',
 365: 'Table Tennis'}

class_map = {
    0: 'Ignor',
    1: 'Clothes',
    2: 'Toys',
    3: 'Makeup',
    4: 'Furniture',
    5: 'Jewelry',
    6: 'Others',
    7: 'Bags',
    8: 'Electrical',
    9: 'Shoes',
    10: 'Accessories',
    11: 'Food',
    12: 'Stationery',
    13: 'Empty'
 }


def render_result(img, bboxes, color=[255, 255, 255]):
    for bbox in bboxes:
        class_idx, x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    for bbox in bboxes:
        class_idx, x1, y1, x2, y2 = bbox
        class_name = class_map[class_idx]
        y_text = y1 + 20
        render_text = "{} - {}".format(class_idx, class_name)
        cv2.putText(img, render_text, (int(x1), int(y_text)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)


def main():
    # args = parse_args()
    save_dir = '/home/liwang/data/product_det/product_det_20221121_10233/images'
    anno_path = '/home/liwang/data/product_det/product_det_20221121_10233/anno_13class.txt'
    lmdb_path = '/home/liwang/data/product_det/product_det_20221121_10233/lmdb'

    os.makedirs(save_dir, exist_ok=True)

    env_db = lmdb.open(lmdb_path, readonly=True, lock=False) 
    txn = env_db.begin()

    annos = load_annotations_label(anno_path)

    anno_list = []
    classes = []
    for it in tqdm(annos):
        name = it[0]
        bboxes = it[1]
        for box in bboxes:
            class_id = box[0]
            area = (box[4] - box[2]) * (box[3] - box[1])
            if class_id not in classes and area > 8000:
                classes.append(class_id)
                anno_list.append((name, [box]))
    

    # embed()

    for name, bboxes in tqdm(anno_list):
        assert len(bboxes) == 1, 'box: {}'.format(bboxes)
        class_idx = bboxes[0][0]
        if class_idx == -1:
            continue
        class_name = class_map[class_idx]

        value = txn.get(name.encode())
        data = bytearray(value)
        img = cv2.imdecode(np.asarray(data, dtype=np.uint8), cv2.IMREAD_COLOR)

        result = bboxes
        render_result(img, result)

        # if class_idx ==102:
        #     embed()

        if ' ' in class_name:
            class_name = class_name.replace(' ', '_')
        if '/' in class_name:
            class_name = class_name.replace('/', '_')
        save_path = os.path.join(save_dir, '{}_{}.jpg'.format(class_idx, class_name))
        cv2.imwrite(save_path, img)



if __name__=='__main__':
    main()