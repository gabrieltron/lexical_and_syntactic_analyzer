from dataclasses import dataclass, field
from lark import Lark, v_args, Visitor, Tree, Token
from typing import Any, Dict, Tuple


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

    num_expression : term opt_term              -> expr_start       // opt_term.left_val = term.val
                                                                    // num_expression.val = opt_term.result
                                                                    // num_expression.type = opt_term.type
    
    opt_term : "+" term opt_term                -> opt_term         // opt_term1.left_val = opt_term.lev_val + term.val
                                                                    // opt_term.result = opt_term1.result
            | "-" term opt_term                 -> opt_term         // same
            |                                   -> opt_term         // opt_term.result = opt_term.left_val

    term : unary_expr unary_vazio               -> term             // term.val = unary_vazio.result
                                                                    // unary_vazio.left_val = unary_expr.val


    unary_expr : factor                         -> unary            // unary_expr.val = factor.val
            | "+" factor                        -> unary            // unary_expr.val = + factor.val
            | "-" factor                        -> unary            // unary_expr.val = - factor.val

    factor : INT_CONSTANT                       -> factor_int           // factor.val = int-constant
                                                                    // factor.type = int
            | STRING_CONSTANT                   -> factor_string           // factor.val = string-constant
                                                                    // factor.type = string
            | "null"                            -> factor_null           // factor.val = 0
                                                                    // factor.type = int
            | IDENT lvalue                      -> factor_ident          // factor.val = lvalue.result
                                                                    // factor.type = lvalue.type
                                                                    // lvalue.pos = symbol_table(ident)
                                                                    // lvalue.type = symbol_table(ident)
                                                                    // lvalue.dimensions = symbol_table(ident)
            | "(" num_expression ")"            -> factor_expr          // factor.val = num_expression.val

    unary_vazio : "*" unary_expr unary_vazio    -> unary_vazio      //unary_vazio1.left_val = unary_vazio.left_val * unary_expr.val
                                                                    //unary_vazio.result = unary_vazio1.result
            | "/" unary_expr unary_vazio        -> unary_vazio      //same
            | "%" unary_expr unary_vazio        -> unary_vazio      //same
            |                                   -> unary_vazio      //unary_vazio.result = unary_vazio.left_val
        
    lvalue : "[" INT_CONSTANT "]" lvalue        -> lvalue           //lvalue1.pos = 
                                                                    //lvalue.result = lvalue1.result 
            |                                   -> lvalue           //lvalue.result = symbol_table(lvalue.pos)
    
    
    %import common.INT -> INT_CONSTANT
    %import common.ESCAPED_STRING -> STRING_CONSTANT
    %import common.WS
    %import common.CNAME -> IDENT
    %ignore WS
''', start='statement', keep_all_tokens=True)




class CalculateTree(Visitor):
    def __init__(self, symbol_table: SymbolTable):
        self.symbol_table = symbol_table

    def __default__(self, *args):
        pass

    def _call_userfunc(self, tree: Tree, attrs: Dict[Tuple[str, str], Any]):
        return getattr(self, tree.data, self.__default__)(tree, attrs)

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
                child_attrs = {key:value for key, value in attrs.items() if key[0] == child.data}
                child_attrs = self.visit(child, child_attrs)
                for key, value in child_attrs.items():
                    name, attr = key
                    while (name, '_exist') in attrs:
                        name += '\''inherit_type
                    attrs[(name, attr)] = value

        self._call_userfunc(tree, attrs)
        my_attributes = {key:value for key, value in attrs.items() if key[0] == tree.data and key[1] != '_exist'}
        return my_attributes

    def expr_start(self, tree, attributes):
        attributes[('opt_term', 'left_val')] = attributes[('term', 'val')]
        attributes[('opt_term', 'left_type')] = attributes[('term', 'type')]
        attributes[('num_expression', 'left_val')] = attributes[('opt_term', 'result')]
        attributes[('num_expression', 'left_type')] = attributes[('opt_term', 'type')]

    def unary(self, tree, attributes):
        tokens = [token for token in tree.children if isinstance(token, Token)]
        if attributes[('factor', 'type')] != 'int':
            attributes[('unary', 'val')] = attributes[('factor','val')]
            attributes[('unary', 'type')] = attributes[('factor', 'type')]
        else:
            if len(tokens) == 2:
                token_value = tree.children[0].value
                if token_value == '-':
                    attributes[('unary', 'val')] = 0 - attributes[('factor','val')]
                else: 
                    attributes[('unary', 'val')] = 0 - attributes[('factor','val')]
                attributes[('unary', 'type')] = attributes[('factor','type')]
            else:
                attributes[('unary', 'val')] = attributes[('factor','val')]
                attributes[('unary', 'type')] = attributes[('factor', 'type')]
        

    def factor_int(self, tree, attributes):
        token_value = tree.children[0].value
        attributes[('factor', 'type')] = 'int'
        attributes[('factor', 'val')] = token_value

    def factor_string(self, tree, attributes):
        token_value = tree.children[0].value
        attributes[('factor', 'type')] = 'string'
        attributes[('factor', 'val')] = token_value

    def factor_null(self, tree, attributes):
        attributes[('factor', 'type')] = 'null'
        attributes[('factor', 'val')] = 0

    def factor_ident(self, tree, attributes):
        //TODO
        token_value = tree.children[0].value
        attributes[('factor', 'type')] = 'string'
        attributes[('factor', 'val')] = token_value

    def factor_expr(self, tree, attributes):
        token_value = tree.children[0].value
        attributes[('factor', 'type')] = attributes[('num_expression', 'type')]
        attributes[('factor', 'val')] = attributes[('num_expression', 'val')]

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


text = "x = 1 + 2"
tree = lark_grammar.parse(text)


types = {
    'int': BaseType('int', 4),     
    'char': BaseType('char', 1),
    'str': BaseType('str', 0)  # str is assumed to have size 0 since we need
                                # to know how many chars it'll have first
    }
symbol_table = SymbolTable(types)
visitor = CalculateTree(symbol_table)
visitor.visit(tree)
print(symbol_table.pretty())
