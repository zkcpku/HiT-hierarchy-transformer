import os
import sys
sys.path.append("../code/")

from config import Transformer_python_Train_Config,basedir,Transformer_python_Train_HiT,Transformer_c1400_Train_Config, Transformer_c1400_Train_HiT

class Transformer_python_Train_Probe_Config(Transformer_python_Train_Config):
    def __init__(self):
        super().__init__()
        self.model_type = 'transformer'
        self.output_dir = basedir + 'HiT/save/codenet/probe/python/' + self.model_type + '/'
        self.save_data_dir = self.output_dir
        self.load_model_path = basedir + 'HiT/save/codenet/python/' + self.model_type + '/'
        self.train_data_file = basedir + 'data/codenet/processed_tokens_with_path/python/trainset.json'
        self.eval_data_file = basedir + 'data/codenet/processed_tokens_with_path/python/devset.json'
        self.test_data_file = basedir + 'data/codenet/processed_tokens_with_path/python/testset.json'
        self.seq_vocab_path = basedir + 'HiT/save/codenet/python/seq_vocab.pkl'
        self.seq_vocab_size = 7000
        self.path_vocab_path = None
        self.path_vocab_size = 1
        self.cls_num = 800
        self.train_batch_size = 32
        self.eval_batch_size = 32
        self.epoch = 100
        self.do_eval_before_probe = False
        self.do_train = True
        self.do_eval = False
        self.do_test = False

        self.block_key_words = "block"


class Transformer_python_Train_HiT_Probe(Transformer_python_Train_HiT):
    def __init__(self):
        super().__init__()
        self.model_type = 'HiT_path_node'
        self.output_dir = basedir + 'HiT/save/codenet/probe/python/' + self.model_type + '/'
        self.train_data_file = basedir + 'data/codenet/processed_tokens_with_path/python/trainset.json'
        self.eval_data_file = basedir + 'data/codenet/processed_tokens_with_path/python/testset.json'
        self.save_data_dir = self.output_dir
        
        self.load_model_path = basedir + 'HiT/save/codenet/python/' + self.model_type + '/'

        self.seq_vocab_path = basedir + 'data/codenet/processed_tokens_with_path/python/token_vocab.pkl'
        
        self.path_vocab_path = basedir + 'data/codenet/processed_tokens_with_path/python/path_vocab.pkl'
        self.path_vocab_size = 300
        self.tokenizer_path = 'split_to_node'
        self.path_nlayers = 2
        self.emb_size = (128, 128)
        self.max_path_len = 10
        self.gradient_accumulation_steps = 2
        self.script = basedir +'HiT/code/run_HiT.py'

        self.epoch = 100
        self.do_eval_before_probe = True
        self.do_train = True
        self.do_eval = False
        self.do_test = False

        self.block_key_words = "block"


class Transformer_c1400_Train_Probe_Config(Transformer_c1400_Train_Config):
    def __init__(self):
        super().__init__()
        self.model_type = 'transformer'
        self.output_dir = basedir + 'HiT/save/codenet/probe/c1400/' + self.model_type + '/'
        self.save_data_dir = self.output_dir
        self.load_model_path = basedir + 'HiT/save/codenet/c1400/' + self.model_type + '/'
        self.train_data_file = basedir + 'data/codenet/processed_tokens_with_path/c1400/trainset.json'
        self.eval_data_file = basedir + 'data/codenet/processed_tokens_with_path/c1400/devset.json'
        self.test_data_file = basedir + 'data/codenet/processed_tokens_with_path/c1400/testset.json'
        self.seq_vocab_path = basedir + 'HiT/save/codenet/c1400/seq_vocab.pkl'
        self.seq_vocab_size = 7000 # c1400 7000 cover 0.9810812107170107
        self.whole_path_vocab_path = basedir + 'HiT/save/codenet/c1400/whole_path_vocab.pkl'
        self.whole_path_vocab_size = 9000 # c1400 9000 cover 0.9302234957566
        self.tokenizer_path = 'split_to_node'
        self.remove_comment = True
        self.max_path_len = 20 # c1400 max_path_len = 83, 20 for 0.9978856286997587
        self.emb_size = 256
        self.nhead = 8
        self.nlayers = 8
        self.nhid = 1024
        self.block_size = 512
        self.path_vocab_path = basedir + 'HiT/save/codenet/c1400/path_node_vocab.pkl'
        self.path_vocab_size = 300 # 260
        self.cls_num = 1400
        self.learning_rate = 1e-3
        self.train_batch_size = 36
        self.eval_batch_size = 36
        self.epoch = 100
        self.do_eval_before_probe = True
        self.do_train = True
        self.do_eval = False
        self.do_test = False

        self.limit_dataset = 40000 

        self.block_key_words = "compound"


class Transformer_c1400_Train_HiT_Probe(Transformer_c1400_Train_HiT):
    def __init__(self):
        super().__init__()
        self.model_type = 'HiT_path_node'
        self.output_dir = basedir + 'HiT/save/codenet/probe/c1400-big/' + self.model_type + '/'
        self.train_data_file = basedir + 'data/codenet/processed_tokens_with_path/c1400/trainset.json'
        self.save_data_dir = self.output_dir

        self.load_model_path = basedir + 'HiT/save/codenet/c1400-big/' + self.model_type + '/'
        self.tokenizer_path = 'split_to_node'
        self.path_nlayers = 2
        self.emb_size = (128, 128)
        self.max_path_len = 20
        self.gradient_accumulation_steps = 5
        self.epoch = 100
        self.do_eval_before_probe = True
        self.do_train = True
        self.do_eval = False
        self.do_test = False

        self.limit_dataset = 40000 
        self.block_key_words = "compound"



all_choose_dict = [Transformer_python_Train_Probe_Config, Transformer_python_Train_HiT_Probe, Transformer_c1400_Train_Probe_Config, Transformer_c1400_Train_HiT_Probe]

all_choose_dict = {k.__name__:k for k in all_choose_dict}


choose_args = sys.argv[-1]
if choose_args in all_choose_dict:
    my_config = all_choose_dict[choose_args]()
else:
    raise ValueError("wrong config for " + str(choose_args.__repr__()))