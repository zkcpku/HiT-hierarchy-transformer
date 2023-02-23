import os
import sys
import json

sys.path.append('/home/removename/workspace/HiT/clone_detection/Clone-detection-POJ-104/code/')
from treesitter2anytree import get_leaf_node_list
from tqdm import tqdm
# leaf_name_list, path_list, label, index

seq_root = '/home/removename/workspace/HiT/clone_detection/dataset/poj104/seq/'
seq_path_root = '/home/removename/workspace/HiT/clone_detection/dataset/poj104/seq_path/'


def generate_seq_path(seq_root, seq_path_root):
    for cls_path in ['train.jsonl','valid.jsonl','test.jsonl']:
        with open(os.path.join(seq_root, cls_path), 'r') as f:
            seqs = f.readlines()
            seqs = [json.loads(seq) for seq in seqs]
            
        outputs = []
        for seq in tqdm(seqs):
            label = seq['label']
            index = seq['index']
            code = seq['code']
            leaf_name_list, path_list = get_leaf_node_list(code, generate_ast=False)
            outputs.append([leaf_name_list, path_list, label, index])
        with open(os.path.join(seq_path_root, cls_path), 'w') as f:
            json.dump(outputs, f)
    
generate_seq_path(seq_root, seq_path_root)

        
