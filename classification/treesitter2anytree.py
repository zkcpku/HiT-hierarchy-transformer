import re
from anytree import AnyNode, RenderTree
from tree_sitter import Language, Parser

def get_python_sitter(build_path='/home/zhangkechi/workspace/utils/treesitter_tools/build/csn_parsers.so', language='python'):
    PY_LANGUAGE = Language(
        build_path, language)
    sitter_parser = Parser()
    sitter_parser.set_language(PY_LANGUAGE)

    return sitter_parser


def converttoany(tree, text, abstract=True):
    nodeid = [0]
    currentnodetype = tree.type
    is_ast_node = tree.is_named
    root = AnyNode(type=currentnodetype, name=text[tree.start_byte:tree.end_byte].decode(
        'utf-8'), is_ast_node=is_ast_node, index=0)

    leaf_node = []

    def traverse(tree, parent=None):  # parent: current anynode for tree
        parent_id = parent.index

        # filter out non-ast-node if needed
        def check_literal_node(node):
            if 'literal' in node.type:
                return True
            if node.type in ['string', 'string_literal']:
                return True
            return False
        if abstract:
            filter_children = [
                child for child in tree.children if child.is_named]
        else:
            if check_literal_node(tree):
                filter_children = []
            else:
                filter_children = tree.children

        if len(filter_children) == 0:
            leaf_node.append(parent)

        for child in filter_children:
            # 每个child都有一个与众不同的id吧？
            childtype = child.type
            is_ast_node = child.is_named
            # if abstract and not is_ast_node:
            #     continue
            nodeid[0] += 1
            childnode = AnyNode(type=childtype, name=text[child.start_byte:child.end_byte].decode(
                'utf-8'), is_ast_node=is_ast_node, parent=parent, index=nodeid[0])
#             print(text[child.start_byte:child.end_byte],'\t',child.start_byte,'\t',child.end_byte)
            traverse(child, parent=childnode)
    traverse(tree, parent=root)
    rst_tree = root
    return rst_tree, leaf_node


def myparse(py_code, generate_ast=True, language='python'):
    py_parser = get_python_sitter(language=language)
    py_tree = py_parser.parse(bytes(py_code, 'utf-8'))
    py_rst_tree, leaf_node = converttoany(
        py_tree.root_node, bytes(py_code, 'utf-8'), abstract=generate_ast)
    return py_rst_tree, leaf_node


def get_leaf_node_list(py_code, generate_ast=True,language='python'):
    py_rst_tree, leaf_nodes = myparse(py_code, generate_ast=generate_ast,language=language)
    rst_list = [(leaf_node.name, get_path_from_leaf_node(leaf_node))
                for leaf_node in leaf_nodes]
    leaf_name_list = [e[0] for e in rst_list]
    path_list = [e[1] for e in rst_list]
    return leaf_name_list, path_list


def render_tree(tree):
    print(RenderTree(tree))


def get_path_from_leaf_node(leaf_node):
    nodes_on_path = []
    now_node = leaf_node
    while now_node:
        nodes_on_path.append(now_node)
        now_node = now_node.parent
    return '|'.join([node.type for node in nodes_on_path[::-1]])


def tokenizer_identifier(name):
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

def anytree2SBT(py_rst_tree, tokenize_identifier = True):
    output = []
    def dfs(node):
        if len(node.children) == 0:
            output.append("(")
            this_node_name = [node.name]
            if tokenize_identifier:
                this_node_name = tokenizer_identifier(node.name)
            output.extend(this_node_name)
            # output.append(node.name)
            output.append(")")
            # output.append(node.name)
            output.extend(this_node_name)
        else:
            output.append("(")
            output.append(node.type)
            for child in node.children:
                dfs(child)
            output.append(")")
            output.append(node.type)
    dfs(py_rst_tree)
    return output


def anytree2XSBT(py_rst_tree,tokenize_identifier = True):
    output = []
    def dfs(node):
        if len(node.children) == 0:
            this_node_name = [node.name]
            if tokenize_identifier:
                this_node_name = tokenizer_identifier(node.name)
            output.extend(this_node_name)
            # output.append(node.name)
        else:
            output.append("<"+node.type+">")
            for child in node.children:
                dfs(child)
            output.append("</"+node.type+">")
    dfs(py_rst_tree)
    return output



def render_SBT(SBT):
    tab_num = 0
    i = 0
    while i < len(SBT):
        if SBT[i] == "(":
            print("\t"*tab_num, SBT[i], SBT[i+1])
            i += 1
            tab_num += 1
        elif SBT[i] == ")":
            tab_num -= 1
            print("\t"*tab_num, SBT[i], SBT[i+1])
            i += 1
        else:
            print("\t"*tab_num, SBT[i])
        i += 1

def render_XSBT(XSBT):
    tab_num = 0
    i = 0
    while i < len(XSBT):
        if XSBT[i].startswith('<') and XSBT[i].endswith('>') and not XSBT[i].startswith('</'):
            print("\t"*tab_num, XSBT[i])
            tab_num += 1
        elif XSBT[i].startswith('</'):
            tab_num -= 1
            print("\t"*tab_num, XSBT[i])
        else:
            print("\t"*tab_num, XSBT[i])
        i += 1



def unit_test(py_code=None,language='python'):
    if py_code is None:
        py_code = '''print('hello')'''
        # import json
        # with open('/home/zhangkechi/workspace/data/codexglue/dataset/java/valid.jsonl', 'r') as f:
        #     py_code = f.readlines()
        #     py_code = py_code[0]
        # py_code = json.loads(py_code)['code']
    print("code:")
    print(py_code)
    print("----------")
    print("AST:")
    ast_tree, ast_leaf_nodes = myparse(py_code, generate_ast=True,language=language)
    render_tree(ast_tree)
    print("all_paths:")
    print([(leaf_node.name, get_path_from_leaf_node(leaf_node))
           for leaf_node in ast_leaf_nodes])

    print("----------")
    print("CST:")
    cst_tree, cst_leaf_nodes = myparse(py_code, generate_ast=False,language=language)
    render_tree(cst_tree)
    print("all_paths:")
    print([(leaf_node.name, get_path_from_leaf_node(leaf_node))
          for leaf_node in cst_leaf_nodes])

    print("----------")
    cst_tree, cst_leaf_nodes = myparse(
           py_code, generate_ast=False, language=language)
    render_tree(cst_tree)
    print(len(cst_leaf_nodes))
    SBT = anytree2SBT(cst_tree)
    print("SBT:", len(SBT))
    # print(anytree2SBT(cst_tree))
    render_SBT(SBT)
    XSBT = anytree2XSBT(cst_tree, False)
    print("XSBT:", len(XSBT))
    render_XSBT(XSBT)


if __name__ == "__main__":
    py_code = '''
def max(a,b):
    if a > b:
        return a
    return b

    '''
    unit_test(py_code,language='python')

