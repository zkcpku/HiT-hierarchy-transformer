import argparse
import os
import sys
import json
import pickle
from torch.utils.data import DataLoader
from dataset import HiTDataset, TextVocab, UniTextVocab, collect_fn, CTTextVocab, VocabEntry, Vocab, PathTokenizer
sys.path.append('/home/REMOVE_NAME/workspace/HiT/methodname/dataset/')
import path_vocab
from path_vocab import *
from trainer import Trainer
from model import Model
import torch
import numpy as np
import random
import datetime


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def train():
    from config import myconfig as args
    if args.seed:
        setup_seed(args.seed_idx)
    print('Experiment on {} dataset'.format(args.dataset))
    writer_path = '{}_{}'.format(args.dataset, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    print(writer_path)
    if args.ct_vocab:
        s_vocab = CTTextVocab(args)
        t_vocab = s_vocab
    else:
        if args.uni_vocab:
            s_vocab = UniTextVocab(args)
            t_vocab = s_vocab
        else:
            s_vocab = TextVocab(args, 'source')
            t_vocab = TextVocab(args, 'target')

    with open(os.path.join(args.dataset, 'path_vocab.pkl'), 'rb') as f:
        path_vocab = pickle.load(f)

    setattr(args, 'seq_vocab_len', len(s_vocab))
    setattr(args, 'path_vocab_len', len(path_vocab))
    setattr(args, 's_vocab', s_vocab)
    setattr(args, 't_vocab', t_vocab)
    setattr(args, 'path_vocab', path_vocab)
    print('seq_vocab_len:', args.seq_vocab_len)
    print('path_vocab_len:', args.path_vocab_len)

    print("Loading Train Dataset")

    if args.data_debug:
        train_dataset = HiTDataset(args, s_vocab, t_vocab, type_=['train','valid'], path_vocab = path_vocab)
    else:
        train_dataset = HiTDataset(args, s_vocab, t_vocab, type_='train', path_vocab = path_vocab)

    print("Loading Valid Dataset")
    valid_dataset = HiTDataset(args, s_vocab, t_vocab, type_='test', path_vocab = path_vocab)

    print("Loading Test Dataset")
    test_dataset = HiTDataset(args, s_vocab, t_vocab, type_='test', path_vocab = path_vocab)

    num_workers = args.num_workers

    print("Creating Dataloader")

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=num_workers,
                                   shuffle=args.shuffle, collate_fn=collect_fn)

    valid_data_loader = DataLoader(valid_dataset, batch_size=args.val_batch_size, num_workers=num_workers,
                                   collate_fn=collect_fn)
    valid_infer_data_loader = DataLoader(valid_dataset, batch_size=args.infer_batch_size, num_workers=num_workers,
                                         collate_fn=collect_fn)
    test_infer_data_loader = DataLoader(test_dataset, batch_size=args.infer_batch_size, num_workers=num_workers,
                                        collate_fn=collect_fn)
    print("Building Model")
    model = Model(args, s_vocab, t_vocab)

    print("Creating Trainer")
    trainer = Trainer(args=args, model=model, train_data=train_data_loader, valid_data=valid_data_loader,
                      valid_infer_data=valid_infer_data_loader, test_infer_data=test_infer_data_loader, t_vocab=t_vocab,
                      writer_path=writer_path)
    if args.load_checkpoint:
        checkpoint_path = args.checkpoint_path
        trainer.load(checkpoint_path)
    print("Training Start")
    for epoch in range(args.epochs):
        if args.train:
            trainer.train(epoch)
        if args.valid:
            trainer.test(epoch)
        test_infer = trainer.predict(epoch, test=False)
        if test_infer:
            trainer.predict(epoch, test=True)
    trainer.writer.close()


if __name__ == '__main__':
    train()
