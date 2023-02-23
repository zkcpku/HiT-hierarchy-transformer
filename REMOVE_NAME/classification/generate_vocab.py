import os, json
import pickle
from tqdm import tqdm
from utils.vocab import VocabEntry


lang = 'python'
data_path = f'/home/REMOVE_NAME/workspace/data/codenet/processed_tokens_with_path/{lang}'
path_out_path = f'/home/REMOVE_NAME/workspace/data/codenet/processed_tokens_with_path/{lang}/path_vocab.pkl'
token_out_path = f'/home/REMOVE_NAME/workspace/data/codenet/processed_tokens_with_path/{lang}/token_vocab.pkl'
path_vocab_size = 300
token_vocab_size = 7000


train_data_path = os.path.join(data_path, 'trainset.json')
with open(train_data_path,'r') as f:
    data = json.load(f)

path_lists = [e[-1] for e in data]
print(f"Example numbers for lang {lang}: {len(path_lists)}.")
all_whole_paths = [e for ee in path_lists for e in ee]
print(f"Path numbers for lang {lang}: {len(all_whole_paths)}.")
path_nodes = [e.split('|') for e in tqdm(all_whole_paths)]
path_node_vocab = VocabEntry.from_corpus(path_nodes, size=path_vocab_size)

print(f"Save path vocab to {path_out_path}.")
with open(path_out_path, 'wb') as f:
    pickle.dump(path_node_vocab, f)


all_code_tokens = [e[-2] for e in data]
seq_vocab = VocabEntry.from_corpus(all_code_tokens, size=token_vocab_size)
print(f"Save token vocab to {token_out_path}.")
with open(token_out_path, 'wb') as f:
    pickle.dump(seq_vocab, f)