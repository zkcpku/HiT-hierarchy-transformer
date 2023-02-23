import numpy as np
import torch
from torch.nn import functional as F

def abs(length):
    return length if length >= 0 else 0


def convert_line(line):
    '''
    convert line into data dict
    :param line: 
    :return: 
    '''
    data = dict()
    target, content, named, paths, paths_map, row, r_path_idx, r_paths = line.strip().split('\t')
    data['target'] = target.split('|')
    data['content'] = content.split('|')
    data['named'] = [int(num) for num in named.split('|')]
    data['paths'] = [[int(num) for num in path.split()] for path in paths.split('|')]
    data['paths_map'] = [[int(num) for num in path_map.split()] for path_map in paths_map.split('|')]
    data['r_path_idx'] = [int(num) for num in r_path_idx.split('|')]
    data['r_paths'] = [[int(num) for num in r_path.split()] for r_path in r_paths.split('|')]
    data['row'] = [int(num) for num in row.split('|')]
    return data


def decoder_process(target, vocab, max_target_len, e_voc=None, pointer=False):
    f_s = [vocab.find(sub_token) for sub_token in target]  # f_s not should been map, because need embedded
    if not pointer:
        f_t = [vocab.find(sub_token) for sub_token in target]
    else:
        assert e_voc is not None
        f_t = []
        for sub_token in target:
            if sub_token in e_voc:
                f_t.append(e_voc[sub_token])
            else:
                f_t.append(
                    vocab.find(sub_token))  # this still exist unk in f_t, cause still some word not in source code
    f_source = [vocab.sos_index] + f_s
    # d_idx = [i + 1 for i in range(len(f_source))][:max_target_len] + [0] * abs(max_target_len - len(f_source))
    f_source = f_source[:max_target_len] + [vocab.pad_index] * abs(max_target_len - len(f_source))
    # f_source: max_target_len
    f_target = f_t + [vocab.eos_index]
    f_target = f_target[:max_target_len] + [vocab.pad_index] * abs(max_target_len - len(f_target))
    # f_target: max_target_len

    return f_source, f_target


def row_process(row, max_code_length):
    min_row = min(row)
    row = [num - min_row + 1 for num in row]  # 0 for padding
    row_ = row[:max_code_length] + [0] * abs(max_code_length - len(row))
    return row_


def content_process(content, vocab, max_code_length, subtokens, e_voc=None, pointer=False):
    # print(content)

    content_ = []
    for tokens in content:
        tokens_ = [vocab.find(token) for token in tokens]
        tokens_ = tokens_[:subtokens] + [vocab.pad_index] * abs(subtokens - len(tokens_))
        content_.append(tokens_)
    if not pointer:
        content_e = None
    else:
        assert e_voc is not None
        content_e = []
        for tokens in content:
            tokens_ = [e_voc[token] if token in e_voc else vocab.find(token) for token in tokens]
            tokens_ = tokens_[:subtokens] + [vocab.pad_index] * abs(subtokens - len(tokens_))
            content_e.append(tokens_)
    padding_tokens = [vocab.pad_index] * subtokens

    content_ = content_[:max_code_length] + [padding_tokens] * abs(max_code_length - len(content))
    if pointer:
        content_e = content_e[:max_code_length] + [padding_tokens] * abs(max_code_length - len(content))
    return content_, content_e

def path_process(paths, path_vocab, max_path_len, max_code_length):
    processed_path_tokens = []
    processed_path_ids = []
    path_tokens = paths
    path_tokens = path_tokens[:max_code_length]
    for each_path in path_tokens:
        each_path_tokens = each_path
        # import pdb;pdb.set_trace()
        # each_path_tokens = tokenizer_path(each_path, args)
        each_path_tokens = each_path_tokens[-max_path_len+2:]
        each_path_tokens = [path_vocab.cls_token] + \
            each_path_tokens+[path_vocab.sep_token]
        each_path_ids = path_vocab.convert_tokens_to_ids(each_path_tokens)
        padding_length = max_path_len - len(each_path_ids)
        each_path_ids += [path_vocab.pad_token_id]*padding_length
        processed_path_ids.append(each_path_ids)
        processed_path_tokens.append(each_path_tokens)
    padding_length = max_code_length - len(processed_path_ids)
    processed_path_ids.extend(
        [[path_vocab.cls_token_id] \
            + [path_vocab.pad_token_id] * (max_path_len - 2) \
            + [path_vocab.sep_token_id]] * padding_length)

    return processed_path_ids



def vnode_action_process(node_actions, vnode=True):
    if not vnode:
        for node_action in node_actions:
            node_action[0] = (0, 0)


def actions_process(node_actions, max_depth, max_ary, max_code_length, vnode=False):
    '''
    需要设置pad idx 为0
    :param node_actions:[[a],[a,b]]
    :param max_depth:
    :param max_ary:
    :return:
    '''
    node_actions = node_actions[:max_code_length]
    overflow_flag = max_ary + 1

    def convert(x):
        # +1 for the overflow flag
        if x > max_ary:
            return overflow_flag
        else:
            return x

    sample_idx = []
    padding_actions = [0] * max_depth
    for node_action in node_actions:
        action_idx = [convert(x) for x in node_action]
        if vnode:
            action_idx[0] = 1
        else:
            action_idx[0] = 0
        sample_idx.append(action_idx[:max_depth] + [0] * abs(max_depth - len(action_idx)))
    return sample_idx[:max_code_length] + [padding_actions] * abs(max_code_length - len(sample_idx))


def pair_common_depth(node_actions, max_code_length):
    '''
    use preprocessed actions to make relative
    :param node_actions: l,depth
    :param max_dia:
    :return:
    '''

    def common_depth(x, y):
        i = 0
        for i in range(min(len(x), len(y))):
            if x[i] != y[i]:
                break
        return i

    node_actions = node_actions[:max_code_length]
    result = []
    for lis_x in node_actions:
        temp_lis = []
        for lis_y in node_actions:
            temp_lis.append(common_depth(lis_x, lis_y))
        result.append(temp_lis[:max_code_length] + [0] * abs(max_code_length - len(temp_lis)))
    return result[:max_code_length] + [[0] * max_code_length] * abs(max_code_length - len(result))


def two_dim_actions_process(node_actions, max_depth, max_ary, max_code_length):
    overflow_flag = max_ary + 1
    sample_idx = []
    padding_actions = [0] * (max_depth * 2)
    for node_action in node_actions:
        action_idx = []
        for x, y in node_action:
            action_idx.append(x if x <= max_ary else overflow_flag)
            action_idx.append(y if y <= max_ary else overflow_flag)
        sample_idx.append(action_idx[:max_depth * 2] + [0] * abs(max_depth * 2 - len(action_idx)))
    return sample_idx[:max_code_length] + [padding_actions] * abs(max_code_length - len(sample_idx))


def commute_time_kernel(A):
    assert np.allclose(A, A.T)
    D = np.diag(np.sum(A, axis=-1, keepdims=False))
    L = D - A
    L_invers = np.linalg.pinv(L)
    return L_invers


def corrected_commute_time_kernel(A):
    assert np.allclose(A, A.T)
    d = np.sum(A, axis=-1, keepdims=False)
    # D = np.diag(d)
    dim = A.shape[0]
    H = np.idenhity(dim) - np.ones(shape=(dim, dim)) / dim
    D_temp = np.diag(np.power(d, -0.5))  # D -1/2
    M_temp = A - np.expand_dims(d, -1) @ np.expand_dims(d, 0) / np.sum(d)  # (A - ddt/vol(G))
    M = D_temp @ M_temp @ D_temp
    return H @ D_temp @ M @ np.linalg.inv(np.idenhity(dim) - M) @ M @ D_temp @ H


def kernel_process(adj, kernel_type):
    if kernel_type == 'Kct':
        kernel = commute_time_kernel(adj)
    elif kernel_type == 'Kcct':
        kernel = corrected_commute_time_kernel(adj)
    else:
        raise Exception('No Valid Kernel Type !!')
    if not torch.is_tensor(kernel):
        return torch.from_numpy(kernel)
    else:
        return kernel


def kernel_pad(kernel, max_code_length):
    # return torch.zeros(max_code_length, max_code_length)

    # print(kernel.dtype)
    l = kernel.shape[0]
    assert l <= max_code_length
    return F.pad(kernel, (0, max_code_length - l, 0, max_code_length - l), "constant", 0).to(torch.int64)


def make_extended_vocabulary(content, vocab):
    e_voc, e_voc_ = dict(), dict()
    idx = len(vocab)
    for tokens in content:
        for token in tokens:
            if not vocab.has_token(token) and token not in e_voc:
                e_voc[token] = idx
                e_voc_[idx] = token
                idx += 1
    return e_voc, e_voc_, idx


def leafPE_process(leaf_idx, max_code_length):
    leaf_idx = leaf_idx[:max_code_length] + [0] * abs(max_code_length - len(leaf_idx))
    return leaf_idx


if __name__ == '__main__':
    from scipy.sparse import csr_matrix

    A = np.array([[0, 1, 0, 0, 1], [1, 0, 1, 1, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 1], [1, 0, 0, 1, 0]])
    S = csr_matrix(A)
    print(kernel_process(S, 'Kct', 8))
    print(commute_time_kernel(A))
