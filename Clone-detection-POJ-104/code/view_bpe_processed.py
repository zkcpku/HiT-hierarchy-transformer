from run import *
import pickle
test_processed_path = '/home/REMOVE_NAME/workspace/HiT/clone_detection/dataset/poj104/seq_path/testset.json_processed'

with open(test_processed_path,'rb') as f:
    test_processed = pickle.load(f)

import pdb;pdb.set_trace()