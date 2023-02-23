import sys

basedir = "/home/removename/workspace/"

class baseModelConfig():
    def __init__(self):
        self.model_type = 'transformer'
        self.output_dir = None
        self.save_data_dir = self.output_dir

        self.load_model_path = None

        self.train_data_file = None
        self.eval_data_file = None
        self.test_data_file = None
        self.seq_vocab_path = None
        self.seq_vocab_size = 7000
        self.path_vocab_path = None
        self.path_vocab_size = 1
        self.cls_num = 800

        self.emb_size = 256
        self.nhead = 8
        self.nhid = 512
        self.nlayers = 6
        self.dropout = 0.2

        self.cache_dir = ''
        self.block_size = 256
        self.do_train = True
        self.do_eval = False
        self.do_test = False
        self.evaluate_during_training = True
        self.do_lower_case = False
        self.train_batch_size = 32
        self.eval_batch_size = 32
        self.gradient_accumulation_steps = 2
        self.learning_rate = 5e-05
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-08
        self.max_grad_norm = 1.0
        self.num_train_epochs = 1.0
        self.max_steps = -1
        self.warmup_steps = 0
        self.logging_steps = 50
        self.save_steps = 50
        self.save_total_limit = None
        self.eval_all_checkpoints = False
        self.no_cuda = False
        self.overwrite_output_dir = False
        self.overwrite_cache = False
        self.seed = 42
        self.epoch = 30
        self.fp16 = False
        self.fp16_opt_level = 'O1'
        self.local_rank = -1
        self.server_ip = ''
        self.server_port = ''


    def print_config(self) -> None:
        for k in self.__dict__:
            print(k, '\t', self.__dict__[k])

    def __repr__(self) -> str:
        return str(self.__dict__)


class Transformer_java_Train_Config(baseModelConfig):
    def __init__(self):
        super().__init__()
        self.model_type = 'transformer'
        self.output_dir = basedir + 'HiT/save/codenet/java/' + self.model_type + '/'
        self.save_data_dir = self.output_dir
        self.load_model_path = None
        self.train_data_file = basedir + 'data/codenet/processed_tokens_with_path/java/trainset.json'
        self.eval_data_file = basedir + 'data/codenet/processed_tokens_with_path/java/devset.json'
        self.test_data_file = basedir + 'data/codenet/processed_tokens_with_path/java/testset.json'
        self.seq_vocab_path = basedir + 'HiT/save/codenet/java/token_vocab.pkl'
        self.seq_vocab_size = 7000
        self.path_vocab_path = None
        self.path_vocab_size = 1
        self.cls_num = 250
        self.train_batch_size = 32
        self.eval_batch_size = 32
        self.epoch = 100
        self.script = basedir + 'HiT/code/run_transformer.py'

class Transformer_java_Train_WholePath(Transformer_java_Train_Config):
    def __init__(self):
        super().__init__()
        self.model_type = 'transformer_wholepath'
        self.output_dir = basedir + 'HiT/save/codenet/java/' + self.model_type + '/'
        self.save_data_dir = self.output_dir
        self.whole_path_vocab_path = basedir + 'HiT/save/codenet/java/whole_path_vocab.pkl'
        self.whole_path_vocab_size = 15000
        self.script = basedir + 'HiT/code/run_whole_path.py'

class Transformer_java_Train_ConcatWholePath(Transformer_java_Train_Config):
    def __init__(self):
        super().__init__()
        self.model_type = 'transformer_concat_wholepath'
        self.output_dir = basedir + 'HiT/save/codenet/java/' + self.model_type + '/'
        self.save_data_dir = self.output_dir
        self.whole_path_vocab_path = basedir + "HiT/save/codenet/java/whole_path_vocab.pkl"
        self.whole_path_vocab_size = 15000
        self.emb_size = (128, 128)
        self.script = basedir + 'HiT/code/run_concat_whole_path.py'

class Transformer_java_Train_HiT(Transformer_java_Train_Config):
    def __init__(self):
        super().__init__()
        self.model_type = 'HiT_path_node'
        self.output_dir = basedir + 'HiT/save/codenet/java/' + self.model_type + '/'
        self.save_data_dir = self.output_dir
        self.path_vocab_path = basedir + 'HiT/save/codenet/java/path_vocab.pkl'
        self.path_vocab_size = 300
        self.tokenizer_path = 'split_to_node'
        self.path_nlayers = 2
        self.emb_size = (128, 128)
        self.max_path_len = 20
        self.gradient_accumulation_steps = 2
        self.script = basedir + 'HiT/code/run_HiT.py'

class Transformer_java_Train_BPE(Transformer_java_Train_HiT):
    def __init__(self):
        super().__init__()
        self.model_type = 'HiT_path_bpe'
        self.output_dir = basedir + 'HiT/save/codenet/java/' + self.model_type + '/'
        self.save_data_dir = self.output_dir
        self.path_vocab_path = basedir + 'HiT/save/codenet/java/path_bpe_vocab.pkl'
        self.path_vocab_size = 500
        self.tokenizer_path = 'split_to_node'
        self.path_nlayers = 2
        self.emb_size = (128, 128)
        self.max_path_len = 20
        self.gradient_accumulation_steps = 2
        self.script = basedir + 'HiT/code/run_HiT.py'

class Transformer_java_Test_HiT(Transformer_java_Train_HiT):
    def __init__(self):
        super().__init__()
        self.load_model_path = self.output_dir + 'checkpoint-best-f1/'
        self.do_train = False
        self.do_eval = True
        self.do_test = True

class Transformer_java_Train_HiT_simply(Transformer_java_Train_HiT):
    def __init__(self):
        super().__init__()
        self.model_type = 'HiT_simply_path_node'
        self.output_dir = basedir + 'HiT/save/codenet/java/' + self.model_type + '/'
        self.save_data_dir = self.output_dir

class Transformer_java_Test_HiT_simply(Transformer_java_Train_HiT_simply):
    def __init__(self):
        super().__init__()
        self.load_model_path = self.output_dir + 'checkpoint-best-f1/'
        self.do_train = False
        self.do_eval = True
        self.do_test = True

class Transformer_java_Train_HiT_Conv1d(Transformer_java_Train_HiT):
    def __init__(self):
        super().__init__()
        self.model_type = 'HiT_Conv1d_path_node'
        self.output_dir = basedir + 'HiT/save/codenet/java/' + self.model_type + '/'
        self.save_data_dir = self.output_dir
        self.max_path_len = 20
        self.window_sizes = [2, 3, 4, 5]
        self.path_nlayers = None

class Transformer_java_Test_HiT_Conv1d(Transformer_java_Train_HiT_Conv1d):
    def __init__(self):
        super().__init__()
        self.load_model_path = self.output_dir + 'checkpoint-best-f1/'
        self.do_train = False
        self.do_eval = True
        self.do_test = True





class Transformer_python_Train_Config(baseModelConfig):
    def __init__(self):
        super().__init__()
        self.model_type = 'transformer'
        self.output_dir = basedir + 'HiT/save/codenet/python/' + self.model_type + '/'
        self.save_data_dir = self.output_dir
        self.load_model_path = None
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
        self.script = basedir + 'HiT/code/run_transformer.py'

class Transformer_python_Train_WholePath(Transformer_python_Train_Config):
    def __init__(self):
        super().__init__()
        self.model_type = 'transformer_wholepath'
        self.output_dir = basedir + 'HiT/save/codenet/python/' + self.model_type + '/'
        self.save_data_dir = self.output_dir
        self.whole_path_vocab_path = basedir + 'HiT/save/codenet/python/whole_path_vocab.pkl'
        self.whole_path_vocab_size = 8000
        self.script = basedir + 'HiT/code/run_whole_path.py'

class Transformer_python_Train_ConcatWholePath(Transformer_python_Train_Config):
    def __init__(self):
        super().__init__()
        self.model_type = 'transformer_concat_wholepath'
        self.output_dir = basedir + 'HiT/save/codenet/python/' + self.model_type + '/'
        self.save_data_dir = self.output_dir
        self.seq_vocab_path = basedir + 'HiT/save/codenet/python/token_vocab.pkl'
        self.whole_path_vocab_path = basedir + 'HiT/save/codenet/python/whole_path_vocab.pkl'
        self.whole_path_vocab_size = 8000
        self.emb_size = (128, 128)
        self.script = basedir + 'HiT/code/run_concat_whole_path.py'

class Transformer_python_Train_HiT(Transformer_python_Train_Config):
    def __init__(self):
        super().__init__()
        self.model_type = 'HiT_path_node'
        self.output_dir = basedir + 'HiT/save/codenet/python/' + self.model_type + '/'
        self.train_data_file = basedir + 'data/codenet/processed_tokens_with_path/python/trainset.json'
        self.eval_data_file = basedir + 'data/codenet/processed_tokens_with_path/python/testset.json'
        self.save_data_dir = self.output_dir
        self.seq_vocab_path = basedir + 'data/codenet/processed_tokens_with_path/python/token_vocab.pkl'
        
        self.path_vocab_path = basedir + 'data/codenet/processed_tokens_with_path/python/path_vocab.pkl'
        self.path_vocab_size = 300
        self.tokenizer_path = 'split_to_node'
        self.path_nlayers = 2
        self.emb_size = (128, 128)
        self.max_path_len = 10
        self.gradient_accumulation_steps = 2
        self.script = basedir +'HiT/code/run_HiT.py'
 
class Transformer_python_Train_BPE(Transformer_python_Train_HiT):
    def __init__(self):
        super().__init__()
        self.model_type = 'HiT_path_bpe'
        self.output_dir = basedir + 'HiT/save/codenet/python/' + self.model_type + '/'
        self.save_data_dir = self.output_dir
        self.path_vocab_path = basedir + 'HiT/save/codenet/python/path_bpe_vocab.pkl'
        self.path_vocab_size = 500
        self.tokenizer_path = 'split_to_node'
        self.path_nlayers = 2
        self.emb_size = (128, 128)
        self.max_path_len = 20
        self.gradient_accumulation_steps = 2
        self.script = basedir + 'HiT/code/run_HiT.py'

class Transformer_python_Test_HiT(Transformer_python_Train_HiT):
    def __init__(self):
        super().__init__()
        self.load_model_path = self.output_dir + 'checkpoint-best-f1/'
        self.do_train = False
        self.do_eval = True
        self.do_test = True

class Transformer_python_Train_HiT_simply(Transformer_python_Train_HiT):
    def __init__(self):
        super().__init__()
        self.model_type = 'HiT_simply_path_node'
        self.output_dir = basedir + 'HiT/save/codenet/python/' + self.model_type + '/'
        self.save_data_dir = self.output_dir

class Transformer_python_Test_HiT_simply(Transformer_python_Train_HiT_simply):
    def __init__(self):
        super().__init__()
        self.load_model_path = self.output_dir + 'checkpoint-best-f1/'
        self.do_train = False
        self.do_eval = True
        self.do_test = True

class Transformer_python_Train_HiT_Conv1d(Transformer_python_Train_HiT):
    def __init__(self):
        super().__init__()
        self.model_type = 'HiT_Conv1d_path_node'
        self.output_dir = basedir + 'HiT/save/codenet/python/' + self.model_type + '/'
        self.save_data_dir = self.output_dir
        self.max_path_len = 20
        self.window_sizes = [2, 3, 4, 5]
        self.path_nlayers = None

class Transformer_python_Test_HiT_Conv1d(Transformer_python_Train_HiT_Conv1d):
    def __init__(self):
        super().__init__()
        self.load_model_path = self.output_dir + 'checkpoint-best-f1/'
        self.do_train = False
        self.do_eval = True
        self.do_test = True






class Transformer_c1000_Train_Config(baseModelConfig):
    def __init__(self):
        super().__init__()
        self.model_type = 'transformer'
        self.output_dir = basedir + 'HiT/save/codenet/c1000/' + self.model_type + '/'
        self.save_data_dir = self.output_dir
        self.load_model_path = None
        self.train_data_file = basedir + 'data/codenet/processed_tokens_with_path/c1000/trainset.json'
        self.eval_data_file = basedir + 'data/codenet/processed_tokens_with_path/c1000/devset.json'
        self.test_data_file = basedir + 'data/codenet/processed_tokens_with_path/c1000/testset.json'
        self.seq_vocab_path = basedir + 'data/codenet/processed_tokens_with_path/c1000/token_vocab.pkl'
        self.seq_vocab_size = 7000 # c1000 6000 cover 0.9801361380218859
        self.whole_path_vocab_path = basedir + 'HiT/save/codenet/c1000/whole_path_vocab.pkl'
        self.whole_path_vocab_size = 8000 # c1000 8000 cover 0.9528658360758334 
        self.tokenizer_path = 'split_to_node'
        self.remove_comment = True
        self.max_path_len = 20 
        self.emb_size = 256
        self.nhead = 8
        self.nlayers = 8
        self.nhid = 1024
        self.block_size = 512
        self.path_vocab_path = basedir + 'data/codenet/processed_tokens_with_path/c1000/path_vocab.pkl'
        self.path_vocab_size = 300
        self.cls_num = 1000
        self.train_batch_size = 36
        self.eval_batch_size = 36
        self.epoch = 100
        self.script = basedir + 'HiT/code/run_transformer.py'

class Transformer_c1000_Train_WholePath(Transformer_c1000_Train_Config):
    def __init__(self):
        super().__init__()
        self.model_type = 'transformer_wholepath'
        self.output_dir = basedir + 'HiT/save/codenet/c1000/' + self.model_type + '/'
        self.save_data_dir = self.output_dir
        self.script = basedir + 'HiT/code/run_whole_path.py'

class Transformer_c1000_Train_ConcatWholePath(Transformer_c1000_Train_Config):
    def __init__(self):
        super().__init__()
        self.model_type = 'transformer_concat_wholepath'
        self.output_dir = basedir + 'HiT/save/codenet/c1000/' + self.model_type + '/'
        self.save_data_dir = self.output_dir
        self.emb_size = (128, 128)
        self.script = basedir + 'HiT/code/run_concat_whole_path.py'

class Transformer_c1000_Train_HiT(Transformer_c1000_Train_Config):
    def __init__(self):
        super().__init__()
        self.model_type = 'HiT_path_node'
        self.output_dir = basedir + 'HiT/save/codenet/c1000-big/' + self.model_type + '/'
        self.train_data_file = basedir + 'data/codenet/processed_tokens_with_path/c1000/trainset.json'
        self.save_data_dir = self.output_dir
        self.tokenizer_path = 'split_to_node'
        self.path_nlayers = 2
        self.cls_num = 1000
        self.emb_size = (128, 128)
        self.max_path_len = 20
        self.gradient_accumulation_steps = 5
        self.script = basedir + 'HiT/code/run_HiT.py'

class Transformer_c1000_Train_HiT_simply(Transformer_c1000_Train_HiT):
    def __init__(self):
        super().__init__()
        self.model_type = 'HiT_simply_path_node'
        self.output_dir = basedir + 'HiT/save/codenet/c1000/' + self.model_type + '/'
        self.save_data_dir = self.output_dir

class Transformer_c1000_Train_HiT_Conv1d(Transformer_c1000_Train_HiT):
    def __init__(self):
        super().__init__()
        self.model_type = 'HiT_Conv1d_path_node'
        self.output_dir = basedir + 'HiT/save/codenet/c1000/' + self.model_type + '/'
        self.save_data_dir = self.output_dir
        self.max_path_len = 20
        self.window_sizes = [2, 3, 4, 5]
        self.path_nlayers = None

class Transformer_c1000_Train_BPE(Transformer_c1000_Train_HiT):
    def __init__(self):
        super().__init__()
        self.model_type = 'HiT_path_bpe'
        self.output_dir = basedir + 'HiT/save/codenet/c1000/' + self.model_type + '/'
        self.save_data_dir = self.output_dir
        self.path_vocab_path = basedir + 'HiT/save/codenet/c1000/path_bpe_vocab.pkl'
        self.path_vocab_size = 500
        self.tokenizer_path = 'split_to_node'
        self.path_nlayers = 2
        self.emb_size = (128, 128)
        self.max_path_len = 20
        self.gradient_accumulation_steps = 5
        self.script = basedir + 'HiT/code/run_HiT.py'

class Transformer_c1000_Test_HiT(Transformer_c1000_Train_HiT):
    def __init__(self):
        super().__init__()
        self.load_model_path = self.output_dir + 'checkpoint-best-f1/'
        self.do_train = False
        self.do_eval = True
        self.do_test = True




class Transformer_c1400_Train_Config(baseModelConfig):
    def __init__(self):
        super().__init__()
        self.model_type = 'transformer'
        self.output_dir = basedir + 'HiT/save/codenet/c1400/' + self.model_type + '/'
        self.save_data_dir = self.output_dir
        self.load_model_path = None
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
        self.script = basedir + 'HiT/code/run_HiT.py'

class Transformer_c1400_Train_WholePath(Transformer_c1400_Train_Config):
    def __init__(self):
        super().__init__()
        self.model_type = 'transformer_wholepath'
        self.output_dir = basedir + 'HiT/save/codenet/c1400/' + self.model_type + '/'
        self.save_data_dir = self.output_dir
        self.script = basedir + 'HiT/code/run_whole_path.py'

class Transformer_c1400_Train_ConcatWholePath(Transformer_c1400_Train_Config):
    def __init__(self):
        super().__init__()
        self.model_type = 'transformer_concat_wholepath'
        self.output_dir = basedir + 'HiT/save/codenet/c1400/' + self.model_type + '/'
        self.save_data_dir = self.output_dir
        self.emb_size = (128, 128)
        self.script = basedir + 'HiT/run_concat_whole_path.py'

class Transformer_c1400_Train_HiT(Transformer_c1400_Train_Config):
    def __init__(self):
        super().__init__()
        self.model_type = 'HiT_path_node'
        self.output_dir = basedir + 'HiT/save/codenet/c1400-big/' + self.model_type + '/'
        self.train_data_file = basedir + 'data/codenet/processed_tokens_with_path/c1400/trainset.json'
        self.save_data_dir = self.output_dir
        self.tokenizer_path = 'split_to_node'
        self.path_nlayers = 2
        self.emb_size = (128, 128)
        self.max_path_len = 20
        self.gradient_accumulation_steps = 5
        self.script = basedir + 'HiT/code/run_HiT.py'

class Transformer_c1400_Train_HiT_simply(Transformer_c1400_Train_HiT):
    def __init__(self):
        super().__init__()
        self.model_type = 'HiT_simply_path_node'
        self.output_dir = basedir + 'HiT/save/codenet/c1400/' + self.model_type + '/'
        self.save_data_dir = self.output_dir

class Transformer_c1400_Train_HiT_Conv1d(Transformer_c1400_Train_HiT):
    def __init__(self):
        super().__init__()
        self.model_type = 'HiT_Conv1d_path_node'
        self.output_dir = basedir + 'HiT/save/codenet/c1400/' + self.model_type + '/'
        self.save_data_dir = self.output_dir
        self.max_path_len = 20
        self.window_sizes = [2, 3, 4, 5]
        self.path_nlayers = None

class Transformer_c1400_Train_BPE(Transformer_c1400_Train_HiT):
    def __init__(self):
        super().__init__()
        self.model_type = 'HiT_path_bpe'
        self.output_dir = basedir + 'HiT/save/codenet/c1400/' + self.model_type + '/'
        self.save_data_dir = self.output_dir
        self.path_vocab_path = basedir + 'HiT/save/codenet/c1400/path_bpe_vocab.pkl'
        self.path_vocab_size = 500
        self.tokenizer_path = 'split_to_node'
        self.path_nlayers = 2
        self.emb_size = (128, 128)
        self.max_path_len = 20
        self.gradient_accumulation_steps = 5
        self.script = basedir + 'HiT/code/run_HiT.py'

class Transformer_c1400_Test_HiT(Transformer_c1400_Train_HiT):
    def __init__(self):
        super().__init__()
        self.load_model_path = self.output_dir + 'checkpoint-best-f1/'
        self.do_train = False
        self.do_eval = True
        self.do_test = True




class Transformer_poj104_Train_Config(baseModelConfig):
    def __init__(self):
        super().__init__()
        self.model_type = 'transformer'
        self.output_dir = basedir + 'HiT/save/codenet/poj104/' + self.model_type + '/'
        self.save_data_dir = self.output_dir
        self.load_model_path = None
        self.train_data_file = basedir + 'data/codenet/processed_tokens_with_path/poj104/trainset.json'
        self.eval_data_file = basedir + 'data/codenet/processed_tokens_with_path/poj104/devset.json'
        self.test_data_file = basedir + 'data/codenet/processed_tokens_with_path/poj104/testset.json'
        self.seq_vocab_path = basedir + 'HiT/save/codenet/poj104/token_vocab.pkl'
        self.seq_vocab_size = 7000
        self.whole_path_vocab_path = basedir + 'HiT/save/codenet/poj104/whole_path_vocab.pkl'
        self.whole_path_vocab_size = 9000
        self.tokenizer_path = 'split_to_node'
        self.remove_comment = True
        self.max_path_len = 20
        self.emb_size = 256
        self.nhead = 8
        self.nlayers = 6
        self.nhid = 512
        self.block_size = 512
        self.path_vocab_path = basedir + 'HiT/save/codenet/poj104/path_vocab.pkl'
        self.path_vocab_size = 300 # 260
        self.cls_num = 104
        self.learning_rate = 1e-3
        self.train_batch_size = 8
        self.eval_batch_size = 8
        self.epoch = 100
        self.script = basedir + 'HiT/code/run_HiT.py'

class Transformer_poj104_Train_HiT(Transformer_poj104_Train_Config):
    def __init__(self):
        super().__init__()
        self.model_type = 'HiT_path_node'
        self.output_dir = basedir + 'HiT/save/codenet/poj104/' + self.model_type + '/'
        self.train_data_file = basedir + 'data/codenet/processed_tokens_with_path/poj104/trainset.json'
        self.save_data_dir = self.output_dir
        self.tokenizer_path = 'split_to_node'
        self.path_nlayers = 2
        self.emb_size = (128, 128)
        self.max_path_len = 20
        self.gradient_accumulation_steps = 8
        self.script = basedir + 'HiT/code/run_HiT.py'

class Transformer_poj104_Train_BPE(Transformer_poj104_Train_HiT):
    def __init__(self):
        super().__init__()
        self.model_type = 'HiT_path_bpe'
        self.output_dir = basedir + 'HiT/save/codenet/poj104/' + self.model_type + '/'
        self.save_data_dir = self.output_dir
        self.path_vocab_path = basedir + 'HiT/save/codenet/poj104/path_bpe_vocab.pkl'
        self.path_vocab_size = 500
        self.tokenizer_path = 'split_to_node'
        self.path_nlayers = 2
        self.emb_size = (128, 128)
        self.max_path_len = 20
        self.gradient_accumulation_steps = 8
        self.script = basedir + 'HiT/code/run_HiT.py'

class Transformer_poj104_Test_HiT(Transformer_poj104_Train_HiT):
    def __init__(self):
        super().__init__()
        self.load_model_path = self.output_dir + 'checkpoint-best-f1/'
        self.do_train = False
        self.do_eval = True
        self.do_test = True



# java
HiT_CodeNet_java_Train_config = [Transformer_java_Train_Config, Transformer_java_Train_HiT, Transformer_java_Train_BPE, \
                                    Transformer_java_Train_HiT_simply, Transformer_java_Train_HiT_Conv1d]
HiT_CodeNet_java_Test_config = [Transformer_java_Test_HiT, Transformer_java_Test_HiT_simply, Transformer_java_Test_HiT_Conv1d]

# python
HiT_CodeNet_python_Train_config = [Transformer_python_Train_Config, Transformer_python_Train_WholePath, Transformer_python_Train_BPE, \
                                    Transformer_python_Train_ConcatWholePath, Transformer_python_Train_HiT, \
                                    Transformer_python_Train_HiT_simply, Transformer_python_Train_HiT_Conv1d]
HiT_CodeNet_python_Test_config = [Transformer_python_Test_HiT, Transformer_python_Test_HiT_simply, Transformer_python_Test_HiT_Conv1d]

# c1000
HiT_CodeNet_c1000_Train_config = [Transformer_c1000_Train_Config, Transformer_c1000_Train_WholePath, Transformer_c1000_Train_BPE, \
                                    Transformer_c1000_Train_ConcatWholePath, Transformer_c1000_Train_HiT, \
                                    Transformer_c1000_Train_HiT_simply, Transformer_c1000_Train_HiT_Conv1d]
HiT_CodeNet_c1000_Test_config = [Transformer_c1000_Test_HiT]

# c1400
HiT_CodeNet_c1400_Train_config = [Transformer_c1400_Train_Config, Transformer_c1400_Train_WholePath, Transformer_c1400_Train_BPE, \
                                    Transformer_c1400_Train_ConcatWholePath, Transformer_c1400_Train_HiT, \
                                    Transformer_c1400_Train_HiT_simply, Transformer_c1400_Train_HiT_Conv1d]
HiT_CodeNet_c1400_Test_config = [Transformer_c1400_Test_HiT]

# poj104
HiT_CodeNet_poj104_Train_config = [Transformer_poj104_Train_Config, Transformer_poj104_Train_HiT, Transformer_poj104_Train_BPE]
HiT_CodeNet_poj104_Test_config = [Transformer_poj104_Test_HiT]

all_choose_dict = HiT_CodeNet_java_Train_config + HiT_CodeNet_java_Test_config \
                    + HiT_CodeNet_python_Train_config + HiT_CodeNet_python_Test_config \
                    + HiT_CodeNet_c1000_Train_config + HiT_CodeNet_c1000_Test_config \
                    + HiT_CodeNet_c1400_Train_config + HiT_CodeNet_c1400_Test_config \
                    + HiT_CodeNet_poj104_Train_config + HiT_CodeNet_poj104_Test_config

all_choose_dict = {k.__name__:k for k in all_choose_dict}


choose_args = sys.argv[-1]
if choose_args in all_choose_dict:
    my_config = all_choose_dict[choose_args]()
else:
    my_config = None
    # raise ValueError("wrong config for " + str(choose_args.__repr__()))
# print(my_config)

# my_config = Transformer_java_Test_HiT_Conv1d()
# print(type(my_config))