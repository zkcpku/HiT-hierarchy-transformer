import re
import numpy as np
import pickle
from tqdm import tqdm
import string
import json
import os
import sys
# train: dev: test = 3:1:1
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def split_train_dev_test(ROOT_PATH, OUTPUT_PATH):
    # ROOT_PATH = '/home/removename/workspace/data/Project_CodeNet_Python800/'
    # OUTPUT_PATH = '/home/removename/workspace/HGT-DGL/data/codenet/python/no_share_subtoken/step1'
    cls_dict = {}
    # cls, content, file_path
    all_data = []
    for cls in tqdm(os.listdir(ROOT_PATH)):
        if cls == '.DS_Store':
            continue

        if cls not in cls_dict:
            cls_dict[cls] = len(cls_dict)

        cls_path = os.path.join(ROOT_PATH, cls)
        for file in os.listdir(cls_path):
            file_path = os.path.join(cls_path, file)
            with open(file_path, 'r') as f:
                content = f.read()
            all_data.append((cls_dict[cls], content, file_path))

    with open(os.path.join(OUTPUT_PATH, 'cls_dict.json'), 'w') as f:
        json.dump(cls_dict, f)

    with open(os.path.join(OUTPUT_PATH, 'all_data.json'), 'w') as f:
        json.dump(all_data, f)

    trainset, devset, testset = [[], [], []]
    for i, (cls, content, file_path) in enumerate(all_data):
        if i % 5 == 3:
            devset.append((cls, content, file_path))
        elif i % 5 == 4:
            testset.append((cls, content, file_path))
        else:
            trainset.append((cls, content, file_path))

    with open(os.path.join(OUTPUT_PATH, 'trainset.json'), 'w') as f:
        json.dump(trainset, f)

    with open(os.path.join(OUTPUT_PATH, 'devset.json'), 'w') as f:
        json.dump(devset, f)

    with open(os.path.join(OUTPUT_PATH, 'testset.json'), 'w') as f:
        json.dump(testset, f)

if __name__ == '__main__':
    ROOT_PATH = '/home/removename/workspace/data/Project_CodeNet_C++1000/'
    OUTPUT_PATH = '/home/removename/workspace/data/codenet/processed_seq/c1000'
    split_train_dev_test(ROOT_PATH, OUTPUT_PATH)
