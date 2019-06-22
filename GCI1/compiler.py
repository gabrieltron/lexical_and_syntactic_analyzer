# João Gabriel Trombeta
# Otto Menegasso Pires
# Mathias
# Wagner Braga dos Santos

from dataclasses import dataclass, field
from lark import Lark, v_args, Visitor, Tree, Token
from typing import Any, Dict, Tuple, List, Set


@dataclass
class EvaluationTree():
    data: str
    left: 'EvaluationTree' = field(default_factory=lambda: None)
    right: 'EvaluationTree' = field(default_factory=lambda: None)


@dataclass
class BaseType():
    name: str
    required_memory: int


@dataclass
class Array():
    base_type: BaseType
    n_dimensions: int
    required_memory: int


@dataclass
class TableEntry():
    type: Any
    value: Any = field(default_factory=lambda: None)


@dataclass
class SymbolTable():
    types: Dict[str, BaseType]
    variables: Dict[str, TableEntry] = field(default_factory=dict)

    def pretty(self) -> str:
        total_memory = 0
        s = ''
        for variable, entry in self.variables.items():
            s += 'Variable ' + variable + ': \n'
            if isinstance(entry.type, Array):
                s += 'Type=Array, Base Type={}, Dimensions={}, Required Memory={}, Value={}'.format(
                    entry.type.base_type.name, entry.type.n_dimensions,
                    entry.type.required_memory, entry.value
                )
            else:
                s += 'Type={}, Required Memory={}, Value={}'.format(
                    entry.type.name, entry.type.required_memory, entry.value
                )
            total_memory += entry.type.required_memory
            s += '\n'
        s += 'The program requires a total of {} bytes of memory'.format(total_memory)
        return s


lark_grammar = Lark('''\
    start : dec | exps

    dec : ctype var_decl          -> inherit_type               // var_decl.og_type = ctype.type

    var_decl : IDENT vector extra_var       -> var_decl         // var_decl.dimension = vector.dimension
                                                                // extra_var.og_type = var_decl.og_type
                                                                // var_decl.required_size = vector.required_size
                                                                // var_decl.dimension = vector.dimension
                                                                // var_decl.ident = ident.lexval
                                                                // update_symbol_table(var_decl.ident, var_decl.dimension)

    extra_var : "," IDENT vector extra_var  -> var_decl         // same as above
                |                           -> var_decl

    ctype : "int"                           -> ctype            // ctype.type = int
            | "string"                      -> ctype            // ctype.type = string

    vector : "[" INT_CONSTANT "]" vector    -> vector           // vector.dimension = vector'.dimension + 1
            |                               -> vector           // vector.required_size = vector'.required_size * int-constant


    exps : term opt_term              -> num_expression

    opt_term : "+" term opt_term                -> opt_term
            | "-" term opt_term                 -> opt_term
            |                                   -> opt_term
    term : unary_expr unary_vazio               -> term
    unary_expr : factor                         -> unary_expr
            | "+" factor                        -> unary_expr
            | "-" factor                        -> unary_expr
    factor : INT_CONSTANT                       -> factor
            | STRING_CONSTANT                   -> factor
            | "null"                            -> factor
            | IDENT lvalue                      -> factor
            | "(" exps ")"            -> factor
    unary_vazio : "*" unary_expr unary_vazio    -> unary_vazio
            | "/" unary_expr unary_vazio        -> unary_vazio
            | "%" unary_expr unary_vazio        -> unary_vazio
            |                                   -> unary_vazio

    lvalue : "[" INT_CONSTANT "]" lvalue        -> lvalue
            |                                   -> lvalue


    %import common.INT -> INT_CONSTANT
    %import common.ESCAPED_STRING -> STRING_CONSTANT
    %import common.WS
    %import common.CNAME -> IDENT
    %ignore WS
''', start='start', keep_all_tokens=True)


class CalculateTree(Visitor):
    def __init__(self, symbol_table: SymbolTable):
        self.symbol_table = symbol_table

    def __default__(self, *args):
        pass

    def _call_userfunc(self, tree: Tree, attrs: Dict[Tuple[str, str], Any]):
        return getattr(self, tree.data, self.__default__)(tree, attrs)

    def print_labels(self, node):
        if isinstance(node, Tree):
            for child in node.children:
                self.print_labels(child)
            print(node.data)
        else:
            print(node.value)

    def visit(self, tree: Tree, attrs: Dict[Tuple[str, str], Any] = {}):
        '''Visits the tree in a bottom up manner, trying to execute father's
        semantic actions before going to children, so we can pass an attribute
        from a left child to a right child.
        Args:
            tree: Current subtree
            attrs: Semantic attributes, they can be from the current node, from
            the parent or from the direct children

        Returns:
            The set of attributes of itself so the father can have access to it
        '''

        attrs[(tree.data, '_exist')] = True
        for child in tree.children:
            try:
                self._call_userfunc(tree, attrs)
            except KeyError:
                pass

            if isinstance(child, Tree):
                child_attrs = {}
                for key, value in attrs.items():
                    if key[0] == child.data:
                        child_attrs[key] = value
                    elif len(key[0]) > len(tree.data) and key[0][:len(tree.data)] == tree.data:
                        child_attrs[(tree.data, key[1])] = value

                child_attrs = self.visit(child, child_attrs)
                for key, value in child_attrs.items():
                    name, attr = key
                    while (name, '_exist') in attrs:
                        name += '\''
                    attrs[(name, attr)] = value

        self._call_userfunc(tree, attrs)
        my_attributes = {key:value for key, value in attrs.items() if key[0] == tree.data and key[1] != '_exist'}
        return my_attributes

    def opt_term(self, tree, attributes):#ok
        if not tree.children:
            return

        root = EvaluationTree(tree.children[0].value)
        left_child = attributes[('opt_term', 'left')]
        root.left = left_child
        righ_child = attributes['term', 'tree']
        root.right = righ_child
        if tree.children[2].children:
            attributes[('opt_term\'', 'left')] = root.right
            root.right = attributes[('opt_term\'', 'tree')]
        attributes[('opt_term', 'tree')] = root

    def lvalue(self, tree, attributes): #ok
        if not tree.children:
            attributes[('lvalue', 'index')] = ''
        else:
            tokens = [token.value for token in tree.children if isinstance(token, Token)]
            attributes[('lvalue', 'index')] = ''.join(tokens) + attributes[('lvalue\'', 'index')]

    def unary_vazio(self, tree, attributes):#ok
        if not tree.children:
            return None

        op = tree.children[0].value
        root = EvaluationTree(op)

        root.left = attributes[('unary_vazio', 'left')]
        operator = attributes[('unary_expr', 'tree')]
        if tree.children[2].children:
            attributes[('unary_vazio\'', 'left')] = operator
            right_tree = attributes[('unary_vazio\'', 'tree')]
            root.right = right_tree
        else:
            root.right = operator

        attributes[('unary_vazio', 'tree')] = root

    def term(self, tree, attributes):#ok
        if tree.children[1].children:
            attributes[('unary_vazio', 'left')] = attributes[('unary_expr', 'tree')]
            final_tree = attributes[('unary_vazio', 'tree')]
        else:
            final_tree = attributes[('unary_expr', 'tree')]

        attributes[('term', 'tree')] = final_tree
        
    def num_expression(self, tree, attributes): #ok
        root = attributes[('term', 'tree')]
        if tree.children[1].children:
            attributes[('opt_term', 'left')] = root

            root = attributes[('opt_term', 'tree')]

        attributes[('num_expression', 'tree')] = root
        self.eval_tree = root

    def unary_expr(self, tree, attributes):#ok
        if len(tree.children) == 2:
            sign = tree.children[0].value
            tree = attributes[('factor', 'tree')]
            tree.data = sign + tree.data
        attributes[('unary_expr', 'tree')] = attributes[('factor', 'tree')]
            
    def factor(self, tree, attributes):#ok
        tokens = [token for token in tree.children if isinstance(token, Token)]

        if len(tokens) == 1:
            token_value = tokens[0].value
            node_name = token_value + attributes[('lvalue', 'index')]
            attributes[('factor', 'tree')] = EvaluationTree(node_name)
        elif tokens[0] == '(':
            attributes[('factor', 'tree')] = attributes[('num_expression', 'tree')]

    def inherit_type(self, tree, attributes):
        attributes[('var_decl', 'og_type')] = attributes[('ctype', 'type')]

    def ctype(self, tree, attributes):
        token_value = tree.children[0].value
        attributes[('ctype', 'type')] = token_value

    def var_decl(self, tree, attributes):
        if not tree.children:
            return None

        tokens = [token for token in tree.children if isinstance(token, Token)]
        id = tokens[0].value if len(tokens) == 1 else tokens[1].value
        attributes[('var_decl', 'ident')] = id
        attributes[('var_decl', 'dimension')] = attributes[('vector','dimension')]
        attributes[('var_decl','n_elements')] = attributes[('vector','n_elements')]
        attributes[('extra_var','og_type')] = attributes[('var_decl','og_type')]
        self.update_symbol_table(id, attributes[('extra_var','og_type')], attributes[('vector','dimension')], attributes[('vector','n_elements')])

    def vector(self, tree, attrs):
        if not tree.children:
            attrs[('vector', 'dimension')] = 0
            attrs[('vector', 'n_elements')] = 1
            return None
        else:
            size = int(tree.children[1].value)
            attrs[('vector'), ('dimension')] = attrs[('vector\''), ('dimension')] + 1
            attrs[('vector'), ('n_elements')] = attrs[('vector\'','n_elements')] * size

    def update_symbol_table(self, ident, _type, dimension = 0, n_elements = 0):
        type = self.symbol_table.types[_type]
        if dimension == 0:
            self.symbol_table.variables[ident] = TableEntry(type)
        else:
            array = Array(type, dimension, n_elements*type.required_memory)
            self.symbol_table.variables[ident] = TableEntry(array)


def three_address_expression(
    tree : EvaluationTree, operators: Set[str], code_list: List[str], counter: List[int] = [0]
    ) -> str:
    '''Create a list recursively of tree addresses code for the given EvaluationTree.
    Args:
        tree: Tree to be evaluated.
        operators: operators present in the grammar, like "+" for sum.
        code_list: List where the codes will be stored
        counter: Count how many op have already been created

    Returns:
        The name of the created operation from this node of the tree
    '''

    if tree.data in operators:
        op1 = three_address_expression(tree.left, operators, code_list, [counter[0]+1])
        op2 = three_address_expression(tree.right, operators, code_list, [counter[0]+1])
        name = 'op' + str(counter[0])
        operation = name + ' = ' + op1 + tree.data + op2
        code_list.append(operation)
        return name
    
    return tree.data


text = input('Insira entrada no formato "EXPS; DEC"\n')
types = {
    'int': BaseType('int', 4),
    'char': BaseType('char', 1),
    'str': BaseType('str', 0)
    }
symbol_table = SymbolTable(types)

exps, decl = text.split('; ')

tree = lark_grammar.parse(exps)
visitor = CalculateTree(symbol_table)
print('Printing three address code')
visitor.visit(tree)
three_address_code = []
three_address_expression(visitor.eval_tree, {'+', '-', '*', '/', '%'}, three_address_code)
for code in three_address_code:
    print(code)
print()

tree = lark_grammar.parse(decl)
visitor = CalculateTree(symbol_table)
print('Required memory')
visitor.visit(tree)
print(symbol_table.pretty())
print()
