# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append("../code/")

import argparse
import logging
import pickle
import random
import json
import numpy as np
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import copy
import multiprocessing
from model.TiT import TiTClf
from model.probe_model import ProbeModel
from probe_config import my_config
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (AdamW, get_linear_schedule_with_warmup)


logger = logging.getLogger(__name__)
CACHE_DICT = {}
RUN_ON_PATH = ['Abs','Rel','Whole'][-1]
print("RUN_ON_PATH:", RUN_ON_PATH)
def split_to_abs_rel(path):
    if "stateme" in path:
        path = path.split("statement|")
        rel_path = path[-1]
        abs_path = 'statement|'.join(path[:-1])
    else:
        path = path.split('|')
        rel_path = path[-1]
        abs_path = '|'.join(path[:-1])
    return abs_path, rel_path


def tokenizer_path(path, args):
    if args.tokenizer_path == 'split_to_node':
        if RUN_ON_PATH == 'Abs':
            path = split_to_abs_rel(path)[0]
        elif RUN_ON_PATH == 'Rel':
            path = split_to_abs_rel(path)[1]
        elif RUN_ON_PATH == 'Whole':
            path = path
        path = path.split('|')
        return path
    else:
        if args.tokenizer_path in CACHE_DICT:
            path_tokenizer = CACHE_DICT[args.tokenizer_path]
        else:
            path_tokenizer = PathTokenizer(json_path=args.tokenizer_path)
            CACHE_DICT[args.tokenizer_path] = path_tokenizer
        path = path_tokenizer.encode(path).tokens
        return path
        # raise ValueError(
        #     "tokenizer_path type not supported with " + str(args.tokenizer_path))


def get_example(item):
    leaf_name_list, path_list, code_vocab, path_vocab, args, label = item
    code_tokens = leaf_name_list
    label = label
    code_vocab = code_vocab
    args = args
    path_tokens = path_list
    path_vocab = path_vocab

    if hasattr(my_config, 'remove_comment') and my_config.remove_comment:
        new_code_tokens = [code_tokens[i] for i in range(len(code_tokens)) if not path_tokens[i].endswith(
            'comment') and not path_tokens[i].endswith('\n')]
        new_path_tokens = [path_tokens[i] for i in range(len(path_tokens)) if not path_tokens[i].endswith(
            'comment') and not path_tokens[i].endswith('\n')]
        code_tokens = new_code_tokens
        path_tokens = new_path_tokens
    if hasattr(my_config, 'remove_define') and my_config.remove_define:
        def filter_f(path):
            cands = ["preproc", "using_declaration"]
            for cand in cands:
                if cand in path:
                    return True
            if path.endswith("\n"):
                return True
            return False
        new_code_tokens = [code_tokens[i] for i in range(
            len(code_tokens)) if not filter_f(path_tokens[i])]
        new_path_tokens = [path_tokens[i] for i in range(
            len(path_tokens)) if not filter_f(path_tokens[i])]
        code_tokens = new_code_tokens
        path_tokens = new_path_tokens
    if len(code_tokens) == 0:
        return None

    return convert_examples_to_features(code_tokens, label, code_vocab, args, path_tokens, path_vocab)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 path_tokens=None,
                 path_ids=None,
                 origin_path_tokens=None

                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label
        self.path_tokens = path_tokens
        self.path_ids = path_ids
        self.origin_path_tokens = origin_path_tokens


def convert_examples_to_features(code_tokens, label, code_vocab, args, path_tokens=None, path_vocab=None):
    #source
    code_tokens = code_tokens[:args.block_size]
    # code_tokens = [code_vocab.cls_token]+code_tokens+[code_vocab.sep_token]
    code_ids = code_vocab.convert_tokens_to_ids(code_tokens)
    padding_length = args.block_size - len(code_ids)
    code_ids += [code_vocab.pad_token_id]*padding_length

    source_tokens = code_tokens
    source_ids = code_ids
    processed_path_tokens = []
    processed_path_ids = []
    
    origin_path_tokens = copy.deepcopy(path_tokens)

    path_tokens = path_tokens[:args.block_size]
    for each_path in path_tokens:
        each_path_tokens = tokenizer_path(each_path, args)
        each_path_tokens = each_path_tokens[-args.max_path_len+2:]
        each_path_tokens = [path_vocab.cls_token] + \
            each_path_tokens+[path_vocab.sep_token]
        each_path_ids = path_vocab.convert_tokens_to_ids(each_path_tokens)
        padding_length = args.max_path_len - len(each_path_ids)
        each_path_ids += [path_vocab.pad_token_id]*padding_length
        processed_path_ids.append(each_path_ids)
        processed_path_tokens.append(each_path_tokens)
    padding_length = args.block_size - len(processed_path_ids)
    processed_path_ids.extend(
        [[path_vocab.cls_token_id] \
            + [path_vocab.pad_token_id] * (args.max_path_len - 2) \
            + [path_vocab.sep_token_id]] * padding_length)

    return InputFeatures(source_tokens, source_ids, label, processed_path_tokens, processed_path_ids, origin_path_tokens)


class TextDataset(Dataset):
    def __init__(self, code_vocab, path_vocab, args, file_path='train', block_size=512, pool=None):
        self.args = args
        self.examples = []
        logger.info("Creating features from index file at %s ", file_path)
        with open(file_path, 'r') as f:
            origin_data = json.load(f)
        data = []
        if hasattr(args, 'limit_dataset'):
            origin_data = origin_data[:args.limit_dataset]
        for item in origin_data:
        # for item in origin_data[:20000]:
            # label, code_src, url, leaf_name_list, path_list
            data.append((item[3], item[4], code_vocab,
                        path_vocab, args, item[0]))
            # leaf_name_list, path_list, code_vocab, path_vocab, args, label

        # if 'test' not in postfix:
        #     data=random.sample(data,int(len(data)*0.1))
        self.examples = pool.map(get_example, tqdm(data, total=len(data)))
        # self.examples = [get_example(item) for item in tqdm(data, total=len(data))]
        self.examples = [ex for ex in self.examples if ex is not None]
        # self.examples = [get_example(item) for item in tqdm(data,total=len(data))]
        if 'train' not in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                for k in example.__dict__.keys():
                    logger.info("%s: %s", k, str(example.__dict__[k]))

    def get_edge_labels(self, input_ids, path_tokens):
        block_path_tokens = [self.args.block_key_words.join(e.split(self.args.block_key_words)[:-1]) if self.args.block_key_words in e else "" for e in path_tokens]
        type_tokens = ["identifier" in e for e in path_tokens]
        edge_labels = torch.full((len(input_ids),len(input_ids)), -100,dtype=torch.long)
        min_len = min(len(input_ids),len(path_tokens))
        for i in range(min_len):
            if type_tokens[i] and block_path_tokens[i]:
                near_flag = True
                for j in range(i+1, min_len):
                    if block_path_tokens[i] != block_path_tokens[j]:
                        near_flag = False
                    if type_tokens[i] and type_tokens[j] and near_flag:
                        edge_labels[i,j] = 1
                    if type_tokens[i] and type_tokens[j] and not near_flag:
                        edge_labels[i,j] = 0
        positive_num = len(edge_labels[edge_labels==1])
        negative_num = len(edge_labels[edge_labels==0])
        if positive_num < negative_num:
            # mask additional negative edges
            probability_matrix = torch.full((negative_num,), (negative_num-positive_num)/negative_num)
            mask_negative_edges = torch.bernoulli(probability_matrix)
            mask_negative_edges *= -100
            # change type
            mask_negative_edges = mask_negative_edges.type(edge_labels.type())
            edge_labels[edge_labels == 0] = mask_negative_edges

        return edge_labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item].input_ids), torch.tensor(self.examples[item].path_ids), torch.tensor(self.examples[item].label), self.get_edge_labels(self.examples[item].input_ids,self.examples[item].origin_path_tokens)


def load_and_cache_examples(args, code_vocab, path_vocab, evaluate=False, test=False, pool=None, cache_tag=None):
    if cache_tag is not None and cache_tag in CACHE_DICT:
        return CACHE_DICT[cache_tag]
    dataset = TextDataset(code_vocab, path_vocab, args, file_path=args.test_data_file if test else (
        args.eval_data_file if evaluate else args.train_data_file), block_size=args.block_size, pool=pool)
    if cache_tag is not None:
        CACHE_DICT[cache_tag] = dataset
    return dataset


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, code_vocab, path_vocab, pool):
    """ Train the model """

    train_sampler = RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size * args.n_gpu)
    args.max_steps = args.epoch*len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * args.n_gpu)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mrr = 0.0
    best_f1 = 0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    # Added here for reproducibility (even between python 2 and 3)
    set_seed(args.seed)

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)
            path_inputs = batch[1].to(args.device)
            # labels = batch[2].to(args.device)
            edge_labels = batch[3].to(args.device)

            model.train()
            loss, __ = model([inputs, path_inputs], edge_labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss/tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(
                    np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:

                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(
                            args, model, code_vocab, path_vocab, pool=pool, eval_when_training=True)
                        # Save model checkpoint
                    if args.local_rank == 0 and args.evaluate_during_training:
                        results = evaluate(
                            args, model, code_vocab, path_vocab, pool=pool, eval_when_training=True)
                        # model_to_save = model.module if hasattr(model, 'module') else model

                    if results['eval_f1'] > best_f1:
                        best_f1 = results['eval_f1']
                        logger.info("  "+"*"*20)
                        logger.info("  Best f1:%s", round(best_f1, 4))
                        logger.info("  "+"*"*20)

                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(
                            args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(
                            model, 'module') else model
                        output_dir = os.path.join(
                            output_dir, '{}'.format('model.bin'))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info(
                            "Saving model checkpoint to %s", output_dir)


        logger.info("Eval at epoch {}".format(idx))
        results = evaluate(args, model, code_vocab, path_vocab, pool=pool, eval_when_training=True)
        logger.info("Eval Result at epoch {}".format(str(results)))

        if args.max_steps > 0 and global_step > args.max_steps:
            # train_iterator.close()
            break
    return global_step, tr_loss / global_step



def evaluate(args, model, code_vocab, path_vocab, prefix="", pool=None, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    cache_tag = 'evaluate'
    if eval_when_training == True:
        cache_tag = 'eval_when_training'
    eval_output_dir = args.output_dir
    eval_dataset = load_and_cache_examples(
        args, code_vocab, path_vocab, evaluate=True, pool=pool, cache_tag=cache_tag)
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(
        eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size * args.n_gpu, num_workers=4, pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size * args.n_gpu)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        path_inputs = batch[1].to(args.device)
        # labels = batch[2].to(args.device)
        edge_labels = batch[3].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model([inputs, path_inputs], edge_labels)
            masked_labels = edge_labels[edge_labels != -100]
            masked_logit = logit[edge_labels != -100]
            eval_loss += lm_loss.mean().item()
            logits.append(masked_logit.cpu().numpy())
            y_trues.append(masked_labels.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = logits > 0.5
    y_preds=y_preds.astype(int)

    recall = recall_score(y_trues, y_preds, average='micro')
    precision = precision_score(y_trues, y_preds, average='micro')
    f1 = f1_score(y_trues, y_preds, average='micro')
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "positive_labels_count": float(np.sum(y_trues)),
        "negative_labels_count": float(np.sum(1-y_trues))
    }

    target_names = ['negative','positive']
    logger.info(classification_report(y_trues,y_preds,target_names = target_names))

    logger.info("***** Eval results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result

def evaluate_origin_model(args, model, code_vocab, path_vocab, prefix="", pool=None, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    cache_tag = 'evaluate'
    if eval_when_training == True:
        cache_tag = 'eval_when_training'
    eval_output_dir = args.output_dir
    eval_dataset = load_and_cache_examples(
        args, code_vocab, path_vocab, evaluate=True, pool=pool, cache_tag=cache_tag)
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(
        eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size * args.n_gpu, num_workers=4, pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size * args.n_gpu)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        path_inputs = batch[1].to(args.device)
        labels = batch[2].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model([inputs, path_inputs], labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = np.argmax(logits, 1)

    recall = recall_score(y_trues, y_preds, average='micro')
    precision = precision_score(y_trues, y_preds, average='micro')
    f1 = f1_score(y_trues, y_preds, average='micro')
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1)
    }

    logger.info("***** Eval results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test(args, model, code_vocab, path_vocab, prefix="", pool=None):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    cache_tag = 'test'
    eval_dataset = load_and_cache_examples(
        args, code_vocab, path_vocab, test=True, pool=pool, cache_tag=cache_tag)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(
        eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size * args.n_gpu, num_workers=4, pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size * args.n_gpu)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        path_inputs = batch[1].to(args.device)
        labels = batch[2].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model([inputs, path_inputs], labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = np.argmax(logits, 1)
    with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
        for example, pred in zip(eval_dataset.examples, y_preds):
            f.write(str(example.label) + '\t' + str(pred) + '\n')

    recall = recall_score(y_trues, y_preds, average='micro')
    precision = precision_score(y_trues, y_preds, average='micro')
    f1 = f1_score(y_trues, y_preds, average='micro')
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1)
    }

    logger.info("***** Eval results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))


def main():
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument("--local_rank", type=int, default=-1,
                            help="For distributed training: local_rank")
    cmd_args, unknown = cmd_parser.parse_known_args()
    # print(unknown)

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count)
    args = my_config
    args.local_rank = cmd_args.local_rank

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Barrier to make sure only the first process in distributed training download model & vocab
        torch.distributed.barrier()

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(
            checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(
            checkpoint_last, args.start_epoch))

    code_vocab, path_vocab = None, None
    if args.seq_vocab_path:
        code_vocab = pickle.load(open(args.seq_vocab_path, 'rb'))
    if args.path_vocab_path:
        path_vocab = pickle.load(open(args.path_vocab_path, 'rb'))

    if args.block_size <= 0:
        args.block_size = 256

    if args.model_type == 'TiT_path_node':
        encoder = TiTClf([args.seq_vocab_size, args.path_vocab_size], args.cls_num, args.emb_size, args.nhead,
                       args.nhid, args.nlayers, path_nlayers=args.path_nlayers, dropout=args.dropout)
        checkpoint_last = os.path.join(args.load_model_path, 'checkpoint-best-f1')
        checkpoint_last = os.path.join(checkpoint_last, 'model.bin')
        encoder.load_state_dict(torch.load(checkpoint_last))
        logger.info("reload model from {}".format(checkpoint_last))
        model = ProbeModel(encoder, sum(args.emb_size))
        model.to(args.device)
    # elif args.model_type == 'TiT_simply_path_node':
    #     model = TiT_simply([args.seq_vocab_size, args.path_vocab_size], args.cls_num, args.emb_size, args.nhead,
    #                        args.nhid, args.nlayers, path_nlayers=args.path_nlayers, dropout=args.dropout)
    # elif args.model_type == 'TiT_Conv1d_path_node':
    #     model = TiT_Conv1d([args.seq_vocab_size, args.path_vocab_size], args.cls_num, args.emb_size, args.nhead,
    #                        args.nhid, args.nlayers, path_len=args.max_path_len, window_sizes=args.window_sizes, dropout=args.dropout)
    else:
        raise ValueError("model type not supported")

    if args.local_rank == 0:
        # End of barrier to make sure only the first process in distributed training download model & vocab
        torch.distributed.barrier()

    logger.info("Training/evaluation parameters %s", args)

    # Evaluation
    results = {}
    results = {}
    if args.do_eval_before_probe and args.local_rank in [-1, 0]:
        result=evaluate_origin_model(args, encoder, code_vocab, path_vocab, pool=pool)
        logger.info("***** Eval results before Probe *****")
        logger.info("Eval Result {}".format(str(results)))


    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache
            torch.distributed.barrier()
        cache_tag = 'train'
        train_dataset = load_and_cache_examples(
            args, code_vocab, path_vocab, evaluate=False, pool=pool, cache_tag=cache_tag)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(
            args, train_dataset, model, code_vocab, path_vocab, pool)

    # # Evaluation
    # results = {}
    # if args.do_eval and args.local_rank in [-1, 0]:
    #     checkpoint_prefix = 'checkpoint-best-f1/model.bin'
    #     output_dir = os.path.join(
    #         args.output_dir, '{}'.format(checkpoint_prefix))
    #     model.load_state_dict(torch.load(output_dir))
    #     model.to(args.device)
    #     result = evaluate(args, model, code_vocab, path_vocab, pool=pool)

    # if args.do_test and args.local_rank in [-1, 0]:
    #     checkpoint_prefix = 'checkpoint-best-f1/model.bin'
    #     output_dir = os.path.join(
    #         args.output_dir, '{}'.format(checkpoint_prefix))
    #     model.load_state_dict(torch.load(output_dir))
    #     model.to(args.device)
    #     test(args, model, code_vocab, path_vocab, pool=pool)

    return results


if __name__ == "__main__":
    main()
