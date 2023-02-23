import os, json
import pickle
from tqdm import tqdm
from utils.vocab import VocabEntry


lang = 'poj'
data_path = f'/home/REMOVE_NAME/workspace/data/codenet/processed_tokens_with_SBT/{lang}'
path_out_path = f'/home/REMOVE_NAME/workspace/data/codenet/processed_tokens_with_SBT/{lang}/XSBT_vocab.pkl'
token_out_path = f'/home/REMOVE_NAME/workspace/data/codenet/processed_tokens_with_SBT/{lang}/SBT_vocab.pkl'
XSBT_vocab_size = 7000
SBT_vocab_size = 7000


train_data_path = os.path.join(data_path, 'trainset.json')
with open(train_data_path,'r') as f:
    data = json.load(f)

path_lists = [e[-1] for e in data]
print(f"Example numbers for lang {lang}: {len(path_lists)}.")
print([e[:5] for e in path_lists[:5]])
path_node_vocab = VocabEntry.from_corpus(path_lists, size=XSBT_vocab_size)

print(f"Save path vocab to {path_out_path}.")
with open(path_out_path, 'wb') as f:
    pickle.dump(path_node_vocab, f)


all_code_tokens = [e[-2] for e in data]
print(f"Example tokens for lang {lang}: {len(all_code_tokens)}.")
print([e[:5] for e in all_code_tokens[:5]])
seq_vocab = VocabEntry.from_corpus(all_code_tokens, size=SBT_vocab_size)
print(f"Save token vocab to {token_out_path}.")
with open(token_out_path, 'wb') as f:
    pickle.dump(seq_vocab, f)
