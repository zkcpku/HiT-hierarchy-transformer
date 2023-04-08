from torch.utils.data import Dataset
import os
import torch
import random
from .process_utils import decoder_process, content_process, make_extended_vocabulary, kernel_process, kernel_pad, \
    actions_process, two_dim_actions_process, vnode_action_process, leafPE_process, path_process
import pickle as pkl
import json
import joblib
import numpy as np
from tqdm import tqdm
import time
import multiprocessing
from .vocab import *
from .path_vocab import *

class HiTDataset(Dataset):
    '''
    # ##
    # data: {'all':['str','str',...,],'adj':[[],[]],'target':[str,str],'mask':[int],'code_tokens':['str','str',...,]}
    # all means the all tokens feed into transformer, including inter_node and terminal
    # note that the inter_node not split into parts
    # adj means the adjacency matrix
    # mask is used to mask the inter_node for pointer network, 1 for code token, 0 for inter node
    # code_tokens for creating poniter ext voc
    # ###
    '''

    def __init__(self, args, s_vocab, t_vocab, type_, path_vocab=None):
        self.dataset_dir = args.dataset
        self.s_vocab = s_vocab
        self.t_vocab = t_vocab
        self.path_vocab = path_vocab
        self.args = args
        self.type_ = type_
        assert type_ in ['train', 'test', 'valid']

        self.json_path = os.path.join(self.dataset_dir, type_ + '.jsonl')
        # self.offset_path = os.path.join(self.dataset_dir, type_ + '_offset.pkl')

        # with open(self.offset_path, 'rb') as f:
        #     self.offset = joblib.load(f)
        # if self.args.tiny_data > 0:
            # self.corpus_line = self.args.tiny_data
        with open(self.json_path, 'r') as f:
                lines = f.readlines()
        self.corpus_line = len(lines)
        if self.args.spl:
            print('Loading in Memory One Process')
            self.data = []
            for line in tqdm(lines):
                self.data.append(self.get_example(line))
        else:
            print('Loading in Memory Multi Process')
            pool = multiprocessing.Pool()
            self.data = pool.map(self.get_example, tqdm(lines))
            pool.close()
            pool.join()

    def __len__(self):
        return self.corpus_line

    def __getitem__(self, item):
        sample = self.data[item]
        return {key: value if torch.is_tensor(value) or isinstance(value, dict) else torch.tensor(value) for key, value
                in sample.items()}
                
    def get_example(self, line):
        return self.process(json.loads(line))

    def process(self, data):
        # import pdb;pdb.set_trace()
        if self.args.pointer:
            assert self.args.uni_vocab, 'separate vocab not support'
            e_voc, e_voc_, voc_len = make_extended_vocabulary(data['input'], self.s_vocab)
        else:
            e_voc, e_voc_, voc_len = None, None, None
        f_source, f_target = decoder_process(data['target'], self.t_vocab, self.args.max_target_len,
                                             e_voc, self.args.pointer)

        content_, content_e = content_process(data['input'], self.s_vocab, self.args.max_code_length,
                                              self.args.subtokens, e_voc, self.args.pointer)

        path_ids = path_process(
            data['path'], self.path_vocab, self.args.max_path_length, self.args.max_code_length)

        # add path
        data_dic = {'f_source': f_source, 'f_target': f_target, 'content': content_, 'path_ids': path_ids}

        if self.args.pointer:
            data_dic['e_voc'] = e_voc
            data_dic['e_voc_'] = e_voc_
            data_dic['voc_len'] = voc_len
            data_dic['content_e'] = content_e

        return data_dic


def collect_fn(batch):
    data = dict()
    max_content_len, max_target_len = 0, 0
    for sample in batch:
        c_l = torch.count_nonzero(torch.sum(sample['content'], dim=-1)).item()
        f_l = torch.count_nonzero(sample['f_source']).item()
        if c_l > max_content_len: max_content_len = c_l
        if f_l > max_target_len: max_target_len = f_l
    data['f_source'] = torch.stack([b['f_source'] for b in batch], dim=0)[:, :max_target_len]
    data['f_target'] = torch.stack([b['f_target'] for b in batch], dim=0)[:, :max_target_len]
    data['content'] = torch.stack([b['content'] for b in batch], dim=0)[:, :max_content_len, :]
    data['path_ids'] = torch.stack([b['path_ids'] for b in batch], dim=0)[:, :max_content_len, :]

    if 'e_voc' in batch[0]:
        data['e_voc'] = [b['e_voc'] for b in batch]
        data['e_voc_'] = [b['e_voc_'] for b in batch]
        max_voc_len = torch.max(torch.stack([b['voc_len'] for b in batch], dim=0)).item()
        data['voc_len'] = torch.tensor(
            [max_voc_len for _ in batch])  # we set e voc len equal for all data in batch, for data parallel
        data['content_e'] = torch.stack([b['content_e'] for b in batch], dim=0)[:, :max_content_len, :]
    return data


if __name__ == '__main__':
    import sys
    import os
    sys.path.append("/home/REMOVE_NAME/workspace/HiT/methodname")
    from config import myconfig as args
    s_vocab = CTTextVocab(args)
    t_vocab = s_vocab
    print(s_vocab)
    print(len(s_vocab))

    with open(os.path.join(args.dataset, 'path_vocab.pkl'), 'rb') as f:
        path_vocab = pkl.load(f)
    print(len(path_vocab))
    
    ruby_dataset = HiTDataset(args, s_vocab, t_vocab, 'test', path_vocab = path_vocab)
    print(ruby_dataset.__gehitem__(0)['path_ids'].shape)

