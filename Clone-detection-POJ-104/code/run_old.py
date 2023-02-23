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
from model_old import Model

from treesitter2anytree import get_leaf_node_list
from config import my_config as args
from vocab import VocabEntry


cpu_cont = multiprocessing.cpu_count()
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

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 path_ids,
                 index,
                 label,
                 full_input_tokens,
                 leaf_name_list, 
                 path_list

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.path_ids = path_ids
        self.index=index
        self.label=label
        self.full_input_tokens=full_input_tokens
        self.leaf_name_list = leaf_name_list
        self.path_list = path_list

        
def convert_examples_to_features(js,tokenizer,args,path_vocab):
    code_src = js['code']
    leaf_name_list, path_list = get_leaf_node_list(
        code_src, generate_ast=False)

    source_leaf_nodes = leaf_name_list
    source_leaf_nodes = [' ' + source_leaf_nodes[i] if i !=
                         0 else source_leaf_nodes[i] for i in range(len(source_leaf_nodes))]
    source_each_tokens = [tokenizer.tokenize(e) for e in source_leaf_nodes]
    source_paths = path_list
    source_each_paths = [[source_paths[i]] *
                     len(source_each_tokens[i]) for i in range(len(source_each_tokens))]
    
    
    

    code_src = ' '.join(code_src.split())
    code=' '.join(leaf_name_list)
    # code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    code_tokens = [e for ee in source_each_tokens for e in ee][:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]

    source_paths = [e for ee in source_each_paths for e in ee][:args.block_size-2]
    source_paths = [path_vocab.cls_token] + source_paths+[path_vocab.sep_token]

    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    path_ids = [path_vocab[e] for e in source_paths]

    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    path_ids+=[path_vocab.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids,path_ids,js['index'],int(js['label']),code_src,leaf_name_list,path_list)

def generate_train_path_lists(file_path=None):
    data = []
    with open(file_path) as f:
        for line in tqdm(f):
            line=line.strip()
            js=json.loads(line)
            code_src = js['code']
            leaf_name_list, path_list = get_leaf_node_list(
                code_src, generate_ast=False)
            data.append(path_list)
    return data

def generate_path_vocab(train_path_lists=None, save=True, load=True):
    if 'path_vocab' in ALL_DATA_EXAMPLE:
        return ALL_DATA_EXAMPLE['path_vocab']
    if load:
        if 'path_vocab.pkl' in os.listdir(args.save_data_dir):
            with open(args.save_data_dir + 'path_vocab.pkl', 'rb') as f:
                path_vocab = pickle.load(f)
            print("[ZKCPKU] loaded ", args.save_data_dir + 'path_vocab.pkl')
            ALL_DATA_EXAMPLE['path_vocab'] = path_vocab

            return path_vocab

    paths = train_path_lists
    path_vocab = VocabEntry.from_corpus(
        paths, size=args.path_vocab_size)
    if save:
        with open(args.save_data_dir + 'path_vocab.pkl', 'wb') as f:
            pickle.dump(path_vocab, f)
        print("[ZKCPKU] saved ", args.save_data_dir + 'path_vocab.pkl')

    ALL_DATA_EXAMPLE['path_vocab'] = path_vocab

    return path_vocab


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, path_vocab, file_path=None):
        self.examples = []
        data=[]
        with open(file_path) as f:
            for line in tqdm(f):
                line=line.strip()
                js=json.loads(line)
                data.append(js)
        for js in tqdm(data):
            self.examples.append(convert_examples_to_features(js,tokenizer,args,path_vocab))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("label: {}".format(example.label))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                    logger.info("path_ids: {}".format(' '.join(map(str, example.path_ids))))
                    logger.info("full_input_tokens: {}".format(
                        str(example.full_input_tokens)))
                    logger.info("leaf_name_list: {}".format(str(example.leaf_name_list)))
                    logger.info("path_list: {}".format(str(example.path_list)))
                    

        self.label_examples={}
        for e in self.examples:
            if e.label not in self.label_examples:
                self.label_examples[e.label]=[]
            self.label_examples[e.label].append(e)
        
    def __len__(self):
        return len(self.examples)

    def __gehitem__(self, i):   
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
def evaluate(args, model, tokenizer,eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    global eval_dataset
    if eval_dataset is None:
        path_vocab = generate_path_vocab(load = True)
        eval_dataset = TextDataset(tokenizer, args,path_vocab,args.eval_data_file)

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

def test(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    path_vocab = generate_path_vocab(load = True)
    eval_dataset = TextDataset(tokenizer, args, path_vocab,args.test_data_file)


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

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels=1
    config.path_vocab_size = args.path_vocab_size

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)    
    else:
        model = model_class(config)

    model=Model(model,config,tokenizer,args)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache
        path_vocab = generate_path_vocab(generate_train_path_lists(args.train_data_file),save = True, load = True)
        train_dataset = TextDataset(tokenizer, args, path_vocab,args.train_data_file)
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
        result=evaluate(args, model, tokenizer)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))
            
    if args.do_test and args.local_rank in [-1, 0]:
        checkpoint_prefix = 'checkpoint-best-map/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir),strict=False)                  
        model.to(args.device)
        test(args, model, tokenizer)

    return results


if __name__ == "__main__":
    main()



