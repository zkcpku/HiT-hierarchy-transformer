import argparse
import os
import sys
import json
import pickle
from torch.utils.data import DataLoader
from dataset import HiTDataset, TextVocab, UniTextVocab, collect_fn, CTTextVocab, VocabEntry, Vocab, PathTokenizer
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
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--dataset", type=str, help="train dataset", default='ruby',
                        choices=['python', 'ruby', 'javascript', 'go'])

    # dataset size
    parser.add_argument("--max_code_length", type=int, default=512, help="")
    parser.add_argument("--subtokens", type=int, default=5, help="")
    parser.add_argument("--max_target_len", type=int, default=7, help=' <eos or sos> + true len of method name')

    # vocab
    parser.add_argument("--uni_vocab", type=boolean_string, default=True,
                        help="source vocab (embedding) = target vocab (embedding)")
    parser.add_argument("--weight_tying", type=boolean_string, default=True,
                        help="right embedding = pre softmax matrix ")
    parser.add_argument("--ct_vocab", type=boolean_string, default=True, help="use code transformer's voc")

    # trainer
    parser.add_argument("--with_cuda", type=boolean_string, default=True, help="training with CUDA: true or false")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate of adam")
    parser.add_argument("--warmup", type=boolean_string, default=False, help="")
    parser.add_argument("--lr_sche", type=boolean_string, default=True, help="")
    parser.add_argument("--eps", type=float, default=1e-8, help="")
    parser.add_argument("--clip", type=float, default=1.0, help="0 is no clip")

    parser.add_argument("--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("--accu_batch_size", type=int, default=128,
                        help="number of real batch_size per step, setup for save gpu memory")
    parser.add_argument("--val_batch_size", type=int, default=128, help="number of batch_size of valid")
    parser.add_argument("--infer_batch_size", type=int, default=128, help="number of batch_size of infer")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--num_workers", type=int, default=8, help="dataloader worker size")
    parser.add_argument("--save", type=boolean_string, default=True, help="whether to save model checkpoint")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="0.1 in transformer paper")
    parser.add_argument("--dropout", type=float, default=0.2, help="0.1 in transformer paper")

    # transformer
    parser.add_argument("--embedding_size", type=int, default=512, help="hidden size of transformer model")
    parser.add_argument("--activation", type=str, default='gelu', help="", choices=['gelu', 'relu'])
    parser.add_argument("--hidden", type=int, default=1024, help="hidden size of transformer model")
    parser.add_argument("--d_ff_fold", type=int, default=2, help="ff_hidden = ff_fold * hidden")
    parser.add_argument("--e_ff_fold", type=int, default=2, help="ff_hidden = ff_fold * hidden")
    parser.add_argument("--layers", type=int, default=3, help="number of layers")
    parser.add_argument("--decoder_layers", type=int, default=1, help="number of decoder layers")
    parser.add_argument("--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("--old_PE", default=False, type=boolean_string, help="")

    # Tree
    parser.add_argument("--max_depth", type=int, default=32, help="")
    parser.add_argument("--max_ary", type=int, default=64, help="")
    parser.add_argument("--action_size", type=int, default=32, help="")
    parser.add_argument("--tree", type=boolean_string, default=True, help="")
    parser.add_argument("--spl", type=boolean_string, default=True, help="")
    parser.add_argument("--two_spl", type=boolean_string, default=False, help="")
    parser.add_argument("--two_spl_type", type=str, choices=['mul', 'add'], default='add', help="")
    parser.add_argument("--vnode", type=boolean_string, default=False, help="")
    # Adavance
    parser.add_argument("--leaf_PE", type=boolean_string, default=False, help="")
    parser.add_argument("--leaf_PE_mask", type=boolean_string, default=False, help="")

    # others
    parser.add_argument("--unk_shift", type=boolean_string, default=True, help="")
    parser.add_argument("--embedding_mul", type=boolean_string, default=True, help="")
    parser.add_argument("--pointer", type=boolean_string, default=True, help="")
    parser.add_argument("--pointer_type", type=str, choices=['mul', 'add'], default='mul', help="")

    # debug
    parser.add_argument("--tiny_data", type=int, default=0, help="only a little data for debug")
    parser.add_argument("--seed", type=boolean_string, default=True, help="fix seed or not")
    parser.add_argument("--seed_idx", type=int, default=0, help="fix seed or not")
    parser.add_argument("--data_debug", type=boolean_string, default=False, help="check over-fit for a litter data")
    parser.add_argument("--train", type=boolean_string, default=True, help="Whether to train or just infer")
    parser.add_argument("--valid", type=boolean_string, default=False, help="Whether to train or just infer")
    parser.add_argument("--load_checkpoint", type=boolean_string, default=False,
                        help="load checkpoint for continue train or infer")
    parser.add_argument("--checkpoint", type=str, default='', help="load checkpoint for continue train or infer")
    parser.add_argument("--shuffle", type=boolean_string, default=True, help="")
    parser.add_argument("--old_calculate", type=boolean_string, default=False, help="")

    args = parser.parse_args()
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

    print("Loading Train Dataset")

    if args.data_debug:
        train_dataset = HiTDataset(args, s_vocab, t_vocab, type_='valid', path_vocab)
    else:
        train_dataset = HiTDataset(args, s_vocab, t_vocab, type_='train', path_vocab)

    print("Loading Valid Dataset")
    valid_dataset = HiTDataset(args, s_vocab, t_vocab, type_='valid',path_vocab)

    print("Loading Test Dataset")
    test_dataset = HiTDataset(args, s_vocab, t_vocab, type_='test',path_vocab)

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
        checkpoint_path = 'checkpoint/{}'.format(args.checkpoint)
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
