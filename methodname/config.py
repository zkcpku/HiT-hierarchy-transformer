class baseconfig():
    def __init__(self):
        self.dataset = 'ruby'
        self.max_code_length = 512
        self.subtokens = 5
        self.max_target_len = 7
        self.uni_vocab = True
        # self.weight_tying = True
        self.ct_vocab = True
        self.with_cuda = True
        self.lr = 1e-04
        self.warmup = False
        self.lr_sche = True
        self.eps = 1e-08
        self.clip = 1.0
        self.batch_size = 64
        self.accu_batch_size = 128
        self.val_batch_size = 16
        self.infer_batch_size = 16
        self.epochs = 50
        self.num_workers = 2
        self.save = True
        self.label_smoothing = 0.1
        self.dropout = 0.2
        self.embedding_size = 512
        self.activation = 'gelu'
        self.hidden = 1024
        self.d_ff_fold = 2
        self.e_ff_fold = 2
        self.layers = 3
        self.decoder_layers = 1
        self.attn_heads = 8
        self.old_PE = False
        self.max_depth = 32
        self.max_ary = 64
        self.action_size = 32
        self.tree = True
        self.spl = True
        self.two_spl = False
        self.two_spl_type = 'add'
        self.vnode = False
        self.leaf_PE = False
        self.leaf_PE_mask = False
        self.unk_shift = True
        self.embedding_mul = True
        self.pointer = True
        self.pointer_type = 'mul'
        self.tiny_data = 0
        self.seed = True
        self.seed_idx = 36
        self.data_debug = False
        self.train = True
        self.valid = False
        self.load_checkpoint = False
        self.checkpoint = ''
        self.shuffle = True
        self.old_calculate = False

    def print_config(self) -> None:
        for k in self.__dict__:
            print(k, '\t', self.__dict__[k])

    def __repr__(self) -> str:
        return str(self.__dict__)


class Myconfig(baseconfig):
    def __init__(self):
        super().__init__()
        # dataset & seq_vocab
        self.config_name = 'debug'
        self.dataset = '/home/REMOVE_NAME/workspace/data/methodname/CSN_ruby/tokenize_seq_path_rel/'
        self.uni_vocab = True
        # self.weight_tying = True

        self.pointer = True
        self.spl = True
        self.max_target_len = 10
        self.max_code_length = 512
        self.subtokens = 7

        # path_vocab
        self.max_path_length = 20

        # encoder
        self.batch_size = 10
        self.val_batch_size = 10
        self.infer_batch_size = 10
        self.embedding_size = 256
        self.path_embedding_size = 256
        self.hidden = 256
        self.concat_hidden = self.hidden + self.path_embedding_size
        self.attn_heads = 16
        self.d_ff_fold  = 4
        self.encoder_layers = 4
        self.decoder_layers = 2
        self.path_encoder_layers = 2
        self.dropout = 0.3
        self.activation = 'gelu'

        self.beam_search = False
        self.beam_size = 10

        self.load_checkpoint = False
        self.checkpoint_path = '/home/REMOVE_NAME/workspace/data/methodname/CSN_ruby/tokenize_seq_path_rel/_2022-04-10-20-39-30_0.pth'

        # self.data_debug = True

        



myconfig = Myconfig()
