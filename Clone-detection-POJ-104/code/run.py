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

import argparse
import glob
import logging
import os
import sys
import pickle
import random
import re
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json
# try:
#     from torch.utils.tensorboard import SummaryWriter
# except:
#     from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from model import Model

from treesitter2anytree import get_leaf_node_list
from config import my_config as args
from config import my_config
# from vocab import VocabEntry
sys.path.append('/home/REMOVE_NAME/workspace/HiT/code/utils/')
from vocab import VocabEntry
from pathvocab import PathTokenizer

from HiT_encoder import *



from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}
ALL_DATA_EXAMPLE = {}
CACHE_DICT = {}

RUN_ON_PATH = ['Abs','Rel','Whole'][1]
print("RUN_ON_PATH:", RUN_ON_PATH)

def split_to_abs_rel(path):
    if "statement|" in path:
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
            logger.info(f"load path bpe: {args.tokenizer_path}")
            CACHE_DICT[args.tokenizer_path] = path_tokenizer
        path = path_tokenizer.encode(path).tokens
        return path
        # raise ValueError(
        #     "tokenizer_path type not supported with " + str(args.tokenizer_path))


def get_example(item):
    leaf_name_list, path_list, code_vocab, path_vocab, args, label, index = item
    code_tokens = leaf_name_list
    label = label
    index = index
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

    return convert_examples_to_features(code_tokens, label, index, code_vocab, args, path_tokens, path_vocab)


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids, 
                 label,
                 index,
                 path_tokens,
                 path_ids
    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.path_ids = path_ids
        self.index=index
        self.label=label
        self.path_tokens = path_tokens


def convert_examples_to_features(code_tokens, label, index, code_vocab, args, path_tokens=None, path_vocab=None):
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
        [[path_vocab.cls_token_id]
            + [path_vocab.pad_token_id] * (args.max_path_len - 2)
            + [path_vocab.sep_token_id]] * padding_length)

    return InputFeatures(source_tokens, source_ids, label,index, processed_path_tokens, processed_path_ids)

class TextDataset(Dataset):
    def __init__(self, code_vocab, path_vocab, args, file_path='train', block_size=512, pool = None):
        self.examples = []
        logger.info("Creating features from index file at %s ", str(file_path))
        if type(file_path) == list:
            origin_data = []
            for each_file_path in file_path:
                with open(each_file_path, 'r') as f:
                    origin_data.extend(json.load(f))
        else:
            with open(file_path, 'r') as f:
                origin_data = json.load(f)
        data = []
        for item in origin_data:
            leaf_name_list, path_list, label, index = item
            label = int(label)
            index = int(index)
            data.append((leaf_name_list, path_list, code_vocab,
                        path_vocab, args, label, index))

        # if 'test' not in postfix:
        #     data=random.sample(data,int(len(data)*0.1))
        # pool = multiprocessing.Pool()
        # self.examples = pool.map(get_example, tqdm(data, total=len(data)))
        # pool.close()
        # pool.join()

        self.examples = [get_example(item) for item in tqdm(data, total=len(data))]
        self.examples = [ex for ex in self.examples if ex is not None]
        if args.save_processed_dataset:
            import pickle
            with open(file_path + '_processed','wb') as f:
                pickle.dump(self.examples,f)
            logger.info(f"save processed data: {file_path + '_processed'}")
        # self.examples = [get_example(item) for item in tqdm(data,total=len(data))]
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                for k in example.__dict__.keys():
                    logger.info("%s: %s", k, str(example.__dict__[k]))
        
        self.label_examples = {}
        for e in self.examples:
            if e.label not in self.label_examples:
                self.label_examples[e.label] = []
            self.label_examples[e.label].append(e)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        label=self.examples[i].label
        index=self.examples[i].index
        labels=list(self.label_examples)
        labels.remove(label)
        while True:
            shuffle_example=random.sample(self.label_examples[label],1)[0]
            if shuffle_example.index!=index:
                p_example=shuffle_example
                break
        n_example=random.sample(self.label_examples[random.sample(labels,1)[0]],1)[0]
        
        return (torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].path_ids),
                torch.tensor(p_example.input_ids), torch.tensor(p_example.path_ids),
                torch.tensor(n_example.input_ids), torch.tensor(n_example.path_ids),
                torch.tensor(label))
            
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size,num_workers=4,pin_memory=True)
    args.max_steps=args.epoch*len( train_dataloader)
    args.save_steps=len( train_dataloader)
    args.warmup_steps=len( train_dataloader)
    args.logging_steps=len( train_dataloader)
    args.num_train_epochs=args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

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
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step = args.start_step
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    best_acc=0.0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    for idx in range(args.start_epoch, int(args.num_train_epochs)): 
        bar = train_dataloader
        tr_num=0
        train_loss=0
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)
            input_paths = batch[1].to(args.device)
            p_inputs = batch[2].to(args.device)
            p_input_paths = batch[3].to(args.device)
            n_inputs = batch[4].to(args.device)
            n_input_paths = batch[5].to(args.device)
            labels = batch[6].to(args.device)
            model.train()
            loss,vec = model(inputs,input_paths,p_inputs,p_input_paths,n_inputs,n_input_paths,labels)


            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
            avg_loss=round(train_loss/tr_num,5)
            if (step+1)% 100==0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,avg_loss))
            #bar.set_description("epoch {} loss {}".format(idx,avg_loss))

                
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb=global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer,eval_when_training=True)
                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value,4))                    
                        # Save model checkpoint
                        tr_num=0
                        train_loss=0
 
                    if results['eval_map']>best_acc:
                        best_acc=results['eval_map']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best map:%s",round(best_acc,4))
                        logger.info("  "+"*"*20)                          
                        
                        checkpoint_prefix = 'checkpoint-best-map'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin'))   
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)


eval_dataset=None
def evaluate(args, model, tokenizer,eval_when_training=False,pool = None):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    global eval_dataset
    if eval_dataset is None:
        # (self, code_vocab, path_vocab, args, file_path='train', block_size=512, pool = POOL)
        eval_dataset = TextDataset(tokenizer[0], tokenizer[1], args,args.eval_data_file, pool = pool)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    vecs=[] 
    labels=[]
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        input_paths = batch[1].to(args.device)
        p_inputs = batch[2].to(args.device)
        p_input_paths = batch[3].to(args.device)
        n_inputs = batch[4].to(args.device)
        n_input_paths = batch[5].to(args.device)
        label = batch[6].to(args.device)
        with torch.no_grad():
            lm_loss,vec = model(inputs,input_paths,p_inputs,p_input_paths,n_inputs,n_input_paths,label)
            eval_loss += lm_loss.mean().item()
            vecs.append(vec.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    vecs=np.concatenate(vecs,0)
    labels=np.concatenate(labels,0)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    scores=np.matmul(vecs,vecs.T)
    dic={}
    for i in range(scores.shape[0]):
        scores[i,i]=-1000000
        if int(labels[i]) not in dic:
            dic[int(labels[i])]=-1
        dic[int(labels[i])]+=1
    sort_ids=np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]
    MAP=[]
    for i in range(scores.shape[0]):
        cont=0
        label=int(labels[i])
        Avep = []
        for j in range(dic[label]):
            index=sort_ids[i,j]
            if int(labels[index])==label:
                Avep.append((len(Avep)+1)/(j+1))
        MAP.append(sum(Avep)/dic[label])
          
    result = {
        "eval_loss": float(perplexity),
        "eval_map":float(np.mean(MAP))
    }


    return result

def test(args, model, tokenizer, pool = None):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    # (self, code_vocab, path_vocab, args, file_path='train', block_size=512, pool = POOL)
    eval_dataset = TextDataset(tokenizer[0], tokenizer[1], args,args.test_data_file, pool = pool)


    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    vecs=[] 
    labels=[]
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)    
        input_paths = batch[1].to(args.device)
        p_inputs = batch[2].to(args.device)
        p_input_paths = batch[3].to(args.device)
        n_inputs = batch[4].to(args.device)
        n_input_paths = batch[5].to(args.device)
        label = batch[6].to(args.device)
        with torch.no_grad():
            lm_loss,vec = model(inputs,input_paths,p_inputs,p_input_paths,n_inputs,n_input_paths,label)
            eval_loss += lm_loss.mean().item()
            vecs.append(vec.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    vecs=np.concatenate(vecs,0)
    labels=np.concatenate(labels,0)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    scores=np.matmul(vecs,vecs.T)
    for i in range(scores.shape[0]):
        scores[i,i]=-1000000
    sort_ids=np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]
    indexs=[]
    for example in eval_dataset.examples:
        indexs.append(example.index)
    with open(os.path.join(args.output_dir,"predictions.jsonl"),'w') as f:
        for index,sort_id in zip(indexs,sort_ids):
            js={}
            js['index']=index
            js['answers']=[]
            for idx in sort_id[:499]:
                js['answers'].append(indexs[int(idx)])
            f.write(json.dumps(js)+'\n')

                        
                        
def main():
    logger.info(args)
    cpu_cont = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_cont)
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size=args.train_batch_size//args.n_gpu
    args.per_gpu_eval_batch_size=args.eval_batch_size//args.n_gpu
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    with open('/home/REMOVE_NAME/workspace/HiT/clone_detection/dataset/poj104/seq_path/token_vocab.pkl','rb') as f:
        token_vocab = pickle.load(f)
    print(len(token_vocab))

    with open(args.path_vocab_path,'rb') as f:
        path_vocab = pickle.load(f)
    print(path_vocab)
    tokenizer = [token_vocab, path_vocab]
    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    # config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                        #   cache_dir=args.cache_dir if args.cache_dir else None)
    # config.num_labels=1
    # config.path_vocab_size = args.path_vocab_size

    # tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
    #                                             do_lower_case=args.do_lower_case,
    #                                             cache_dir=args.cache_dir if args.cache_dir else None)
    # if args.block_size <= 0:
    #     args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    # args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    # if args.model_name_or_path:
    #     model = model_class.from_pretrained(args.model_name_or_path,
    #                                         from_tf=bool('.ckpt' in args.model_name_or_path),
    #                                         config=config,
    #                                         cache_dir=args.cache_dir if args.cache_dir else None)    
    # else:
    #     model = model_class(config)
    model = HiTEncoder(ntokens = args.ntokens, emsizes = args.emsizes, nhead = 8, nhid = 2048, nlayers = args.nlayers, path_nlayers=2, dropout=0.3)

    model=Model(model)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    if args.train_load_model:
        checkpoint_prefix = 'checkpoint-best-map/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir),strict=False)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache
        # (self, code_vocab, path_vocab, args, file_path='train', block_size=512, pool = POOL)
        train_dataset = TextDataset(
            tokenizer[0], tokenizer[1], args, args.train_data_file,pool = pool)
        if args.local_rank == 0:
            torch.distributed.barrier()

        train(args, train_dataset, model, tokenizer)



    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoint_prefix = 'checkpoint-best-map/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir),strict=False)      
        model.to(args.device)
        result=evaluate(args, model, tokenizer, pool = pool)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))
            
    if args.do_test and args.local_rank in [-1, 0]:
        checkpoint_prefix = 'checkpoint-best-map/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir),strict=False)                  
        model.to(args.device)
        test(args, model, tokenizer,pool = pool)

    return results


if __name__ == "__main__":
    main()



