import os
import sys
sys.path.append('/home/removename/workspace/utils/treesitter_tools/')
from tqdm import tqdm
from treesitter2anytree import myparse, get_leaf_node_list
import json
import re

TARGET = 'ruby'
lang = TARGET

root_dir = f'/home/removename/workspace/data/methodname/CSN_{TARGET}/{TARGET}/final/jsonl'
target_dir = f'/home/removename/workspace/data/methodname/CSN_{TARGET}/processed_seq_path/'

type_list = ['test','train','valid']



identifier_type = {
    'python': ['identifier', 'list_splat_pattern', 'type_conversion'],
    'ruby': ['identifier', 'hash_key_symbol', 'simple_symbol', 'constant', 'instance_variable', 'global_variable',
             'class_variable'],
    'javascript': ['identifier', 'hash_key_symbol', 'simple_symbol', 'constant', 'instance_variable', 'global_variable',
                   'class_variable', 'property_identifier', 'shorthand_property_identifier', 'statement_identifier',
                   'shorthand_property_identifier_pattern', 'regex_flags'],
    'go': ['identifier', 'hash_key_symbol', 'simple_symbol', 'constant', 'instance_variable', 'global_variable',
           'class_variable', 'property_identifier', 'shorthand_property_identifier', 'statement_identifier',
           'shorthand_property_identifier_pattern', 'regex_flags', 'type_identifier', 'field_identifier',
           'package_identifier', 'label_name']
}
string_type = {
    'python': ['heredoc_content', 'string', 'comment', 'string_literal', 'character_literal', 'chained_string',
               'escape_sequence'],
    'ruby': ['heredoc_content', 'string', 'comment', 'string_literal', 'character_literal', 'chained_string',
             'escape_sequence', 'string_content', 'heredoc_beginning', 'heredoc_end'],
    'javascript': ['heredoc_content', 'string', 'comment', 'string_literal', 'character_literal', 'chained_string',
                   'escape_sequence', 'string_content', 'heredoc_beginning', 'heredoc_end', 'jsx_text',
                   'regex_pattern', 'string_fragment'],
    'go': ['heredoc_content', 'string', 'comment', 'string_literal', 'character_literal', 'chained_string',
           'escape_sequence', 'string_content', 'heredoc_beginning', 'heredoc_end', 'regex_pattern', '\n',
           'raw_string_literal', 'rune_literal']
}


def is_number(s: str):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False
num_type = ['decimal_integer_literal',
            'decimal_floating_point_literal',
            'hex_integer_literal', 'integer',
            'float', 'int_literal', 'imaginary_literal', 'float_literal']

def check_treesitter_error(path_list):
    for path in path_list:
        if 'ERROR' in path:
            return False
    return True
    

def tokenizer(name):
    def camel_case_split(identifier):
        matches = re.finditer(
            '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
            identifier,
        )
        return [m.group(0) for m in matches]
    blocks = []
    for underscore_block in re.split(' |_', name):
        blocks.extend(camel_case_split(underscore_block))
    return blocks


for type_ in type_list:
    all_files = os.listdir(os.path.join(root_dir, type_))
    all_lines = []
    for file in all_files:
        with open(os.path.join(root_dir, type_, file), 'r') as f:
            lines = f.readlines()
            lines = [json.loads(line) for line in lines]
            all_lines.extend(lines)
            del lines
    
    print(type_, len(all_lines))
    # all_lines = all_lines[:1000]
    processed_all_lines = [get_leaf_node_list(
        code_src['original_string'], generate_ast=False, language=lang) for code_src in tqdm(all_lines)]
    index = 0
    err = 0
    methodname_err = 0
    rst = []
    pbar = tqdm(total=len(processed_all_lines))
    for line in processed_all_lines:
        index += 1
        pbar.update(1)
        if not check_treesitter_error(line[1]):
            err += 1
            pbar.set_description(f'{type_} {err}')
            continue
        if TARGET in ['python', 'ruby']:
            if not line[1][1].endswith('identifier'):
                err += 1
                methodname_err += 1
                pbar.set_description(f'{type_} {err}')
                # print(all_lines[index-1])
                # assert methodname_err < 3
                continue
            this_dict = {}
            this_dict['input'] = [tokenizer(name) for name in line[0]]
            this_dict['input'][1] = ['<METHOD>']
            this_dict['path'] = line[1]
            this_dict['target'] = tokenizer(line[0][1])
            for c_index in range(len(this_dict['input'])):
                if len(this_dict['input'][c_index]) == 1 and is_number(this_dict['input'][c_index][0]):
                    this_dict['input'][c_index] = ['<NUM>']
            for p_index in range(len(this_dict['path'])):
                for num_t in num_type:
                    if this_dict['path'][p_index].endswith(num_t):
                        this_dict['input'][p_index] = ['<NUM>']
                        break
            for p_index in range(len(this_dict['path'])):
                for string_t in string_type[TARGET]:
                    if this_dict['path'][p_index].endswith(string_t):
                        this_dict['input'][p_index] = ['<STR>']
                        break
            # print(this_dict)
            rst.append(this_dict)
        elif TARGET in ['go']:
            func_name = all_lines[index-1]['func_name']
            if func_name not in line[0]:
                err += 1
                methodname_err += 1
                pbar.set_description(f'{type_} {err}')
                print(all_lines[index-1])
                assert methodname_err < 3
                continue
            method_name_loc = line[0].index(func_name)
            if not line[1][method_name_loc].endswith('identifier'):
                err += 1
                methodname_err += 1
                pbar.set_description(f'{type_} {err}')
                print(all_lines[index-1])
                assert methodname_err < 3
                continue
            this_dict = {}
            this_dict['input'] = [tokenizer(name) for name in line[0]]
            this_dict['input'][method_name_loc] = ['METHOD']
            this_dict['path'] = line[1]
            this_dict['target'] = tokenizer(line[0][method_name_loc])
            # print(this_dict)
            rst.append(this_dict)
            # print(rst)
            # print(all_lines[index-1])
            # assert False

        else:
            print(line)
            assert False
        # assert False
    print(err)
    with open(os.path.join(target_dir, type_ + '.jsonl'), 'w') as f:
        for line in rst:
            f.write(json.dumps(line) + '\n')

