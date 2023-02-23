import os
import sys
sys.path.append('/home/removename/workspace/treesitter_tools/')
from tqdm import tqdm
from treesitter2anytree import myparse, get_leaf_node_list
import json

data_path = '/home/removename/workspace/data/codenet/processed_seq/c1400/'
new_data_path = '/home/removename/workspace/data/codenet/processed_tokens_with_path/c1400/'
lang = 'cpp'
cls_data = ['trainset','testset','devset']
for e in cls_data:
    print(e)
    this_data_path = data_path + e + '.json'
    with open(this_data_path, 'r') as f:
        data = json.load(f)
    new_data = []
    for i in tqdm(range(len(data))):
        this_data = list(data[i])
        code_src = this_data[1]
        leaf_name_list, path_list = get_leaf_node_list(code_src, generate_ast=False, language=lang)
        # import pdb; pdb.set_trace()
        this_data.append(leaf_name_list)
        this_data.append(path_list)
        
        new_data.append(this_data)
    this_new_data_path = new_data_path + e + '.json'
    with open(this_new_data_path, 'w') as f:
        json.dump(new_data, f)
    print('done' + e)

