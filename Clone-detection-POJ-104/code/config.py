class baseModelConfig():
    def __init__(self):
        self.train_data_file = None
        self.output_dir = None
        self.eval_data_file = None
        self.test_data_file = None
        self.model_type = "bert"
        self.model_name_or_path = None
        self.mlm = False
        self.mlm_probability = 0.15
        self.config_name = ""
        self.tokenizer_name = ""
        self.cache_dir = ""
        self.block_size = -1
        self.do_train = False
        self.do_eval = False
        self.do_test = False
        self.evaluate_during_training = False
        self.do_lower_case = False
        self.train_batch_size = 4
        self.eval_batch_size = 4
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
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
        self.epoch = 42
        self.fp16 = False
        self.fp16_opt_level = 'O1'
        self.local_rank = -1
        self.server_ip = ''
        self.server_port = ''
        self.save_data_dir = './saved/poj/'
        self.path_vocab_size = 50000
        self.remove_comment = False
        self.remove_define = False
        self.tokenizer_path = 'split_to_node'
        self.max_path_len = 15

        # 7978889 230468
        # singlton: 85994
        # 0.9548808111003926


    def print_config(self) -> None:
        for k in self.__dict__:
            print(k, '\t', self.__dict__[k])

    def __repr__(self) -> str:
        return str(self.__dict__)


class TrainConfig(baseModelConfig):
    def __init__(self):
        super().__init__()
        self.train_data_file = '/home/removename/workspace/HiT/clone_detection/dataset/poj104/seq_path/trainset.json'
        self.eval_data_file = '/home/removename/workspace/HiT/clone_detection/dataset/poj104/seq_path/devset.json'
        # self.eval_data_file = '/home/removename/workspace/HiT/clone_detection/dataset/poj104/seq_path/testset.json'
        self.test_data_file = '/home/removename/workspace/HiT/clone_detection/dataset/poj104/seq_path/testset.json'
        self.model_type = 'roberta'
        self.model_name_or_path = 'microsoft/codebert-bases'
        self.mlm = False
        self.mlm_probability = 0.15
        self.config_name = 'microsoft/codebert-base'
        self.tokenizer_name = 'roberta-base'
        self.cache_dir = ''
        self.block_size = 400
        self.do_train = True
        self.do_eval = False
        self.do_test = False
        self.evaluate_during_training = True
        self.do_lower_case = False
        # self.train_batch_size = 16
        # self.eval_batch_size = 16
        self.gradient_accumulation_steps = 1
        self.learning_rate = 2e-05
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
        self.seed = 123456
        self.epoch = 300
        self.fp16 = False
        self.fp16_opt_level = 'O1'
        self.local_rank = -1
        self.server_ip = ''
        self.server_port = ''
        
        self.save_processed_dataset = False

        self.train_load_model = False
        self.train_batch_size = 16
        self.eval_batch_size = 16
        self.tokenizer_path = '/home/removename/workspace/HiT/code/utils/pathvocab.json'
        self.path_vocab_path = '/home/removename/workspace/HiT/clone_detection/dataset/poj104/seq_path/path_bpe_vocab.pkl'
        self.output_dir = './saved_models_bpe'
        self.emsizes = (256,256)
        self.ntokens = (7005,1005)
        self.max_path_len = 20

        # self.tokenizer_path = 'split_to_node'
        # self.path_vocab_path = '/home/removename/workspace/HiT/clone_detection/dataset/poj104/seq_path/path_vocab.pkl'
        # self.output_dir = './saved_models'
        # self.emsizes = (448,64)
        # self.ntokens = (7005, 305)
        # self.max_path_len = 15

class TestConfig(baseModelConfig):
    def __init__(self):
        super().__init__()
        self.train_data_file = '/home/removename/workspace/HiT/clone_detection/dataset/poj104/seq_path/trainset.json'
        self.output_dir = './saved_models'
        self.eval_data_file = '/home/removename/workspace/HiT/clone_detection/dataset/poj104/seq_path/devset.json'
        self.test_data_file = '/home/removename/workspace/HiT/clone_detection/dataset/poj104/seq_path/testset.json'
        self.model_type = 'roberta'
        self.model_name_or_path = 'microsoft/codebert-base'
        self.mlm = False
        self.mlm_probability = 0.15
        self.config_name = 'microsoft/codebert-base'
        self.tokenizer_name = 'roberta-base'
        self.cache_dir = ''
        self.block_size = 400
        self.do_train = False
        self.do_eval = True
        self.do_test = True
        self.evaluate_during_training = True
        self.do_lower_case = False
        self.train_batch_size = 16
        self.eval_batch_size = 16
        self.gradient_accumulation_steps = 1
        self.learning_rate = 2e-05
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
        self.seed = 123456
        self.epoch = 30
        self.fp16 = False
        self.fp16_opt_level = 'O1'
        self.local_rank = -1
        self.server_ip = ''
        self.server_port = ''


my_config = TrainConfig()
# my_config = TestConfig()
print(type(my_config))
