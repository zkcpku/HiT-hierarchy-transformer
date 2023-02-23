import os
import json

PAD, UNK, EOS, SOS = '<PAD>', '<UNK>', '<EOS>', '<SOS>'
ROOT_DIR = None

class Vocab(object):
    def __init__(self, args):
        self.args = args
        self.vocab = dict()
        self.re_vocab = dict()
        self.__special__()

    def find(self, sub_token):
        return self.vocab.get(sub_token, self.unk_index)

    def has_token(self, token):
        if token in self.vocab:
            return True
        else:
            return False

    def has_idx(self, idx):
        if idx in self.re_vocab:
            return True
        else:
            return False

    def __special__(self):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.vocab[PAD] = self.pad_index
        self.vocab[UNK] = self.unk_index
        self.vocab[EOS] = self.eos_index
        self.vocab[SOS] = self.sos_index
        self.special_index = [self.pad_index, self.eos_index, self.sos_index, self.unk_index]

    def re_find(self, idx):
        return self.re_vocab.get(idx, UNK)

    def __len__(self):
        return len(self.vocab)


class CTTextVocab(Vocab):
    def __init__(self, args):
        super().__init__(args)
        self.type = 'CT'
        print('Get CT Vocab')
        self.dataset_dir = args.dataset
        with open(os.path.join(self.dataset_dir, 'ct_vocab.json'), 'r') as f:
            vocab_dict = json.load(f)
        for key, _ in vocab_dict.items():
            if not self.has_token(key):
                self.vocab[key] = len(self.vocab)
        print('{} Vocab length == {}'.format(self.type, len(self.vocab)))
        for key, value in self.vocab.items():
            self.re_vocab[value] = key


class UniTextVocab(Vocab):
    def __init__(self, args):
        '''
        :param args: json dict:{str:num}
        self.vocab: {str:idx}
        :param type: source or target
        '''
        super().__init__(args)
        self.type = 'uni'
        print('Get Uni Vocab')
        self.dataset_dir = os.path.join('./data', args.dataset)
        with open(os.path.join(self.dataset_dir, 'source' + '_vocab.json'), 'r') as f:
            source_vocab_dict = json.load(f)
        with open(os.path.join(self.dataset_dir, 'target' + '_vocab.json'), 'r') as f:
            target_vocab_dict = json.load(f)
        all_vocab_dict = dict()
        for key, value in source_vocab_dict.items():
            if key not in all_vocab_dict:
                all_vocab_dict[key] = value
            else:
                all_vocab_dict[key] += value

        for key, value in target_vocab_dict.items():
            if key not in all_vocab_dict:
                all_vocab_dict[key] = value
            else:
                all_vocab_dict[key] += value

        ordered_list = sorted(all_vocab_dict.items(), key=lambda item: item[1], reverse=True)

        for key, value in ordered_list:
            if value < self.args.vocab_threshold:
                break
            self.vocab[key] = len(self.vocab)
        print('{} Vocab length == {}'.format(self.type, len(self.vocab)))
        self.re_vocab = dict()
        for key, value in self.vocab.items():
            self.re_vocab[value] = key


class TextVocab(Vocab):
    def __init__(self, args, type_):
        '''
        :param args: json dict:{str:num}
        self.vocab: {str:idx}
        :param type_: source or target
        '''
        super().__init__(args)
        assert type_ in ['source', 'target']
        print('Get {} Vocab'.format(type_))
        self.dataset_dir = os.path.join('./data', args.dataset)
        with open(os.path.join(self.dataset_dir, type_ + '_vocab.json'), 'r') as f:
            vocab_dict = json.load(f)
        self.sum_tokens = 0
        for key, value in vocab_dict.items():
            self.sum_tokens += value
        ordered_list = sorted(vocab_dict.items(), key=lambda item: item[1], reverse=True)
        temp_sum = 0
        self.type = type_
        if self.type == 'source':
            self.vocab_portion = self.args.s_vocab_portion
        else:
            self.vocab_portion = self.args.t_vocab_portion
        for key, value in ordered_list:
            temp_sum += value
            if temp_sum / self.sum_tokens > self.vocab_portion:
                break
            self.vocab[key] = len(self.vocab)
        print('{} Vocab length == {}'.format(self.type, len(self.vocab)))
        self.re_vocab = dict()
        for key, value in self.vocab.items():
            self.re_vocab[value] = key
