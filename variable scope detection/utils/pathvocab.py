import json
import random
from tokenizers import Tokenizer
from tokenizers.models import BPE


class PathTokenizer:
    def __init__(self, json_path):
        self.tokenizer = Tokenizer.from_file(json_path)
        self.unk_token = '[UNK]'
        self.unk_token_id = 0
        self.cls_token = '[CLS]'
        self.cls_token_id = 1
        self.sep_token = '[SEP]'
        self.sep_token_id = 2
        self.pad_token = '[PAD]'
        self.pad_token_id = 3
        self.mask_token = '[MASK]'
        self.mask_token_id = 4
        self.special_tokens = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]']
        self.special_end = 'Ġ'

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def __len__(self):
        return self.get_vocab_size()

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def encode(self, seq):
        if seq in self.special_tokens:
            return self.tokenizer.encode(seq)
        else:
            return self.tokenizer.encode(seq+self.special_end)

    def pad_max_length(self, l, max_length=5):
        # 得到后几个字符
        if len(l) >= max_length:
            return l[-max_length:]
        else:
            return l + [self.pad_token] * (max_length - len(l))

    def encode_batch(self, source_paths, max_length=5):
        source_path_tokens = [self.encode(e).tokens for e in source_paths]
        source_path_tokens = [self.pad_max_length(
            e, max_length=max_length) for e in source_path_tokens]
        return source_path_tokens

    def convert_tokens_to_ids(self, path_tokens):
        if type(path_tokens) == type([]):
            rst = [self.tokenizer.token_to_id(e) for e in path_tokens]
            rst = [self.unk_token_id if e == None else e for e in rst]
            return rst
        else:
            rst = self.tokenizer.token_to_id(path_tokens)
            rst = self.unk_token_id if rst == None else rst
            return rst


if __name__ == '__main__':
    path_tokenizer = PathTokenizer(json_path='/home/zhangkechi/workspace/TiT/code/utils/pathvocab(2).json')
    # print(path_tokenizer.get_vocab_size())
    with open('/home/zhangkechi/workspace/data/codenet/processed_tokens_with_path/java/devset.json') as f:
        java_dev_json = json.load(f)
    
    # print(java_dev_json[0][-1][:10])
    # print(path_tokenizer.encode_batch(java_dev_json[0][-1][:20], max_length=10))
    print(path_tokenizer.encode(java_dev_json[0][-1][-1]).tokens)
