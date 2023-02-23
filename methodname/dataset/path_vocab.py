from __future__ import print_function
from collections import Counter
from itertools import chain

import json
import random
from tokenizers import Tokenizer
from tokenizers.models import BPE

class VocabEntry(object):
    def __init__(self):
        self.word2id = dict()
        self.unk_id = 3
        self.word2id['<pad>'] = 0
        self.word2id['<s>'] = 1
        self.word2id['</s>'] = 2
        self.word2id['<unk>'] = 3

        self.cls_token = '<s>'
        self.cls_token_id = 1
        self.sep_token = '</s>'
        self.sep_token_id = 2
        self.pad_token = '<pad>'
        self.pad_token_id = 0
        self.unk_token = '<unk>'
        self.unk_token_id = 3

        self.id2word = {v: k for k, v in self.word2id.items()}

    def __gehitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __sehitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def is_unk(self, word):
        return word not in self

    def to_dict(self):
        return self.word2id

    def convert_tokens_to_ids(self, tokens):
        return [self[token] for token in tokens]

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=0):
        vocab_entry = VocabEntry()

        word_freq = Counter(chain(*corpus))
        non_singletons = [w for w in word_freq if word_freq[w] > 1]
        singletons = [w for w in word_freq if word_freq[w] == 1]
        print('number of word types: %d, number of word types w/ frequency > 1: %d' % (len(word_freq),
                                                                                       len(non_singletons)))
        # print('singletons: %s' % singletons)
        print('number of singletons: %d' % len(singletons))

        top_k_words = sorted(word_freq.keys(), reverse=True,
                             key=word_freq.get)[:size]
        words_not_included = []
        for word in top_k_words:
            if len(vocab_entry) < size:
                if word_freq[word] >= freq_cutoff:
                    vocab_entry.add(word)
                else:
                    words_not_included.append(word)

        # print('word types not included: %s' % words_not_included)
        print('number of word types not included: %d' %
              len(words_not_included))

        return vocab_entry


class Vocab(object):
    def __init__(self, **kwargs):
        self.entries = []
        for key, item in kwargs.items():
            assert isinstance(item, VocabEntry)
            self.__setattr__(key, item)
            self.entries.append(key)

    def __repr__(self):
        return 'Vocab(%s)' % (', '.join('%s %swords' % (entry, getattr(self, entry)) for entry in self.entries))


# if __name__ == '__main__':
#     corpus = [['Human', ':', 'What', 'do', 'we', 'want', '?'],
#               ['Computer', ':', 'Natural', 'language', 'processing']]
#     vocab = VocabEntry.from_corpus(corpus, size=20)
#     print(vocab)
#     print(vocab.to_dict())


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
    path_tokenizer = PathTokenizer(json_path='/home/REMOVE_NAME/workspace/HiT/code/utils/pathvocab(2).json')
    # print(path_tokenizer.get_vocab_size())
    with open('/home/REMOVE_NAME/workspace/data/codenet/processed_tokens_with_path/java/devset.json') as f:
        java_dev_json = json.load(f)
    
    # print(java_dev_json[0][-1][:10])
    # print(path_tokenizer.encode_batch(java_dev_json[0][-1][:20], max_length=10))
    print(path_tokenizer.encode(java_dev_json[0][-1][-1]).tokens)
