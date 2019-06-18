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

    atrib_stat : IDENT lvalue "=" num_expression      -> atrib

    num_expression : term opt_term              -> num_expression  
    
    opt_term : "+" term opt_term                -> opt_term        
            | "-" term opt_term                 -> opt_term        
            |                                   -> opt_term        

    term : unary_expr unary_vazio               -> term             

    unary_expr : factor                         -> unary            
            | "+" factor                        -> unary           
            | "-" factor                        -> unary           

    factor : INT_CONSTANT                       -> factor_int                 
            | STRING_CONSTANT                   -> factor_string                   
            | "null"                            -> factor_null         
            | IDENT lvalue                      -> factor_ident          
            | "(" num_expression ")"            -> factor_expr         

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
''', start='atrib_stat', keep_all_tokens=True)

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
                        name += '\''
                    attrs[(name, attr)] = value

        self._call_userfunc(tree, attrs)
        my_attributes = {key:value for key, value in attrs.items() if key[0] == tree.data and key[1] != '_exist'}
        return my_attributes

    def atrib(self, tree, attributes):
        ident = tree.children[0].value

        attributes[('lvalue', 'ident')] = ident
        attributes[('lvalue', 'position')] = 0

        right_val = attributes[('num_expression', 'val')]
        right_type = attributes[('num_expression', 'type')]

        left_val = attributes[('lvalue', 'val')]
        left_type = attributes[('lvalue', 'type')]

        if left_type != right_type:
            print("Type Exception: trying to attribute a " + right_type + " to a " + left_type + " variable")
        else:
            self.update_table_value(ident, left_type, right_val)

    def opt_term(self, tree, attributes):#ok
        if not tree.children:
            attributes[('opt_term', 'val')] = attributes[('opt_term', 'left_value')]
            attributes[('opt_term', 'type')] = attributes[('opt_term', 'left_type')]
        else:
            op = tree.children[0].value
            right_val = attributes[('term','value')]
            right_type = attributes[('term','type')]
            left_val = attributes[('opt_term'), ('left_val')] 
            left_type = attributes[('opt_term'), ('left_type')] 
            if (right_type != left_type):
                print("Type error exception: expression with values of different types")
            else:
                if op == "+":
                    result = left_val + right_val
                else:
                    if (left_type == 'str'):
                        print("Operator Exception: Invalid operator '-'  for type string")
                        result = ''
                    result = left_value - right_value
                attributes[('opt_term\''), ('left_val')] = result
                attributes[('opt_term\''), ('left_type')] = left_type
                attributes[('opt_term'), ('val')] = attributes[('opt_term\''), ('val')]
                attributes[('opt_term'), ('type')] = attributes[('opt_term\'','type')]  
           
    def lvalue(self, tree, attributes): #ok
        if not tree.children:
            ident = attributes[('lvalue', 'ident')]
            var = self.symbol_table.variables[ident]
            if isinstance(var.type, Array):
                pos = attributes[('lvalue', 'position')]
                if (pos * var.type.base_type.required_memory > var.type.required_memory):
                    #error
                    print("Index out of bounds Exception")
                attributes[('lvalue', 'val')] = var.value[pos]
                attributes[('lvalue', 'type')] = var.type.base_type.name
            else:
                attributes[('lvalue', 'val')] = var.value
                attributes[('lvalue', 'type')] = var.type
        else:
            pos = int(tree.children[1].value)
            attributes[('lvalue\''), ('ident')] = attributes[('lvalue'), ('ident')]
            attributes[('lvalue\''), ('position')] = pos + attributes[('lvalue'), ('position')] # * dimension size
            attributes[('lvalue'), ('val')] = attributes[('lvalue\''), ('val')]
            attributes[('lvalue'), ('type')] = attributes[('lvalue\'','type')]  

    def unary_vazio(self, tree, attributes):#ok
        if not tree.children:
            attributes[('unary_vazio', 'val')] = attributes[('unary_vazio', 'left_value')]
            attributes[('unary_vazio', 'type')] = attributes[('unary_vazio', 'left_type')]
        else:
            op = tree.children[0].value
            right_val = attributes[('unary_expr','value')]
            right_type = attributes[('unary_expr','type')]
            left_val = attributes[('unary_vazio'), ('left_val')] 
            left_type = attributes[('unary_vazio'), ('left_type')] 
            if (right_type != 'int' or left_type != 'int'):
                print("Type error exception: this expression do not support the types given")
            else:
                if op == "*":
                    result = left_val * right_val
                elif op == "/":
                    result = left_value // right_value
                elif op == "%":
                    result = left_value % right_value
                attributes[('unary_vazio\''), ('left_val')] = result
                attributes[('unary_vazio\''), ('left_type')] = left_type
                attributes[('unary_vazio'), ('val')] = attributes[('unary_vazio\''), ('val')]
                attributes[('unary_vazio'), ('type')] = attributes[('unary_vazio\'','type')]
        
    def term(self, tree, attributes):#ok
        attributes[('unary_vazio', 'left_val')] = attributes[('unary_expr', 'val')]
        attributes[('unary_vazio', 'left_type')] = attributes[('unary_expr', 'type')]
        attributes[('term', 'val')] = attributes[('unary_vazio', 'val')]
        attributes[('term', 'type')] = attributes[('unary_vazio', 'type')]

    def num_expression(self, tree, attributes): #ok
        attributes[('opt_term', 'left_val')] = attributes[('term', 'val')]
        attributes[('opt_term', 'left_type')] = attributes[('term', 'type')]
        attributes[('num_expression', 'val')] = attributes[('opt_term', 'val')]
        attributes[('num_expression', 'type')] = attributes[('opt_term', 'type')]

    def unary(self, tree, attributes):#ok
        print(attributes)
        tokens = [token for token in tree.children if isinstance(token, Token)]
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
        
    def factor_int(self, tree, attributes):#ok
        token_value = tree.children[0].value
        attributes[('factor', 'type')] = 'int'
        attributes[('factor', 'val')] = token_value


    def factor_string(self, tree, attributes): #ok
        token_value = tree.children[0].value
        attributes[('factor', 'type')] = 'str'
        attributes[('factor', 'val')] = token_value

    def factor_null(self, tree, attributes): #ok
        attributes[('factor', 'type')] = 'int'
        attributes[('factor', 'val')] = 0

    def factor_ident(self, tree, attributes):#ok
        ident = tree.children[0].value
        attributes[('lvalue', 'ident')] = ident
        attributes[('lvalue', 'position')] = 0
        attributes[('factor', 'val')] = attributes[('lvalue', 'val')]
        attributes[('factor', 'type')] = attributes[('factor', 'type')]

    def factor_expr(self, tree, attributes):# ok
        attributes[('factor', 'type')] = attributes[('num_expression', 'type')]
        attributes[('factor', 'val')] = attributes[('num_expression', 'val')]

    def update_table_value(self, ident, _type, value):# ok
        type = self.symbol_table.types[_type]
        atual = self.symbol_table.variables[ident]
        if type == atual.type:
            self.symbol_table.variables[ident] = TableEntry(type, value)
        else:
            #throw some error
            pass
text = "x = 1 + 2"
tree = lark_grammar.parse(text)


types = {
    'int': BaseType('int', 4),     
    'char': BaseType('char', 1),
    'str': BaseType('str', 0)
    }

variables = {
    'x': TableEntry('int', 0)
}
symbol_table = SymbolTable(types, variables)
visitor = CalculateTree(symbol_table)
visitor.visit(tree)
print(symbol_table.pretty())