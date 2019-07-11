# Joao Gabriel Trombeta
# Otto Menegasso Pires
# Mathias
# Wagner Braga dos Santos

from copy import deepcopy
from dataclasses import dataclass, field
from lark import Lark, v_args, Visitor, Tree, Token
from typing import Any, Dict, Tuple, List, Set


@dataclass
class EvaluationTree():
    data: str
    left: 'EvaluationTree' = field(default_factory=lambda: None)
    right: 'EvaluationTree' = field(default_factory=lambda: None)

    def three_address_code(self):
        if not self.left and not self.right:
            return ['op0 = ' + self.data]
        else:
            code = []
            self._three_address_expression({'+','-','/','*','%','>','>=','<','<=', '==', '!='}, code)
            return code

    def _three_address_expression(
        self, operators: Set[str], code_list: List[str], counter: List[int] = [0]
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

        if self.data in operators:
            counter[0] += 1
            op1 = self.left._three_address_expression(operators, code_list, counter)
            counter[0] += 1
            op2 = self.right._three_address_expression(operators, code_list, counter)
            name = 'op' + str(counter[0])
            operation = name + ' = ' + op1 + self.data + op2
            code_list.append(operation)
            counter[0] -= 1
            counter[0] -= 1
            return name

        return self.data


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

    statement : dec ";"                 -> statement
            | IDENT  var_or_atrib ";"   -> statement
            | print_stat ";"            -> statement
            | read_stat ";"             -> statement
            | if_stat 	                -> statement
            | for_stat 	                -> statement
            | return_stat ";"           -> statement
            | "{" stat_list "}"         -> statement
            | "break" ";"               -> statement
            | ";"                       -> statement

    dec.2 : ctype var_decl  -> inherit_type

    var_decl : IDENT vector extra_var       -> var_decl         // var_decl.dimension = vector.dimension
                                                                // extra_var.og_type = var_decl.og_type
                                                                // var_decl.required_size = vector.required_size
                                                                // var_decl.dimension = vector.dimension
                                                                // var_decl.ident = ident.lexval
                                                                // update_symbol_table(var_decl.ident, var_decl.dimension)

    extra_var : "," IDENT vector extra_var  -> extra_var         // same as above
                |                           

    var_or_atrib.1 : var_decl   -> var_or_atrib
            | atrib_stat	-> var_or_atrib

    atrib_stat : lvalue "=" expression  -> atrib_stat

    atrib_statd : lvalue "=" expression  -> atrib_statd

    print_stat.2 : "print" expression  -> print_stat

    read_stat.2 : "read" IDENT lvalue  -> read_stat

    if_stat.2 : "if" "(" expression ")" statement else    -> if_stat

    else : "else" statement     -> else_stat
            |	                  -> else_stat

    for_stat.2 : "for" "(" IDENT atrib_stat ";" expression ";" IDENT atrib_statd ")" statement     -> for_stat

    return_stat.2 : "return" expressionq  -> return_stat

    stat_list : statement stat_list     -> stat_list
            |		                     -> stat_list

    expression : num_expression opt_expression  -> expression

    expression2 : "," expression expression2 
            | 	

    expressionq : expression 	-> expressionq
            |					 -> expressionq

    arg_list : expression expression2 	
            | 				

    arg_listq : "(" arg_list ")" 	
            |			

    opt_expression : ">" num_expression -> opt_expression	
            | "<"  num_expression       -> opt_expression     	
            | "<=" num_expression       -> opt_expression     	
            | ">=" num_expression       -> opt_expression    	
            | "==" num_expression       -> opt_expression     	
            | "!=" num_expression   -> opt_expression
            |      	

    opt_term : "+" term opt_term                -> opt_term
            |  "-" term opt_term                -> opt_term
            |                                   -> opt_term

    num_expression : term opt_term              -> num_expression

    unary_expr : factor                         -> unary_expr
            | "+" factor                        -> unary_expr
            | "-" factor                        -> unary_expr

    unary_vazio : "*" unary_expr unary_vazio    -> unary_vazio
            | "/" unary_expr unary_vazio        -> unary_vazio
            | "%" unary_expr unary_vazio        -> unary_vazio
            |                                   -> unary_vazio
    term : unary_expr unary_vazio               -> term

    factor : INT_CONSTANT                       -> factor
            | STRING_CONSTANT                   -> factor
            | "null"                            -> factor
            | IDENT lvalue                      -> factor
            | "(" exps ")"                      -> factor

    ctype : "int"                               -> ctype            // ctype.type = int
            | "string"                          -> ctype            // ctype.type = string


    lvalue : "[" INT_CONSTANT "]" lvalue        -> lvalue
            | "." IDENT arg_list? lvalue 	    
            |                                   -> lvalue


    vector : "[" INT_CONSTANT "]" vector        -> vector           // vector.dimension = vector'.dimension + 1
            |                                   -> vector           // vector.required_size = vector'.required_size * int-constant

    exps : term opt_term                        -> num_expression

    %import common.INT -> INT_CONSTANT
    %import common.ESCAPED_STRING -> STRING_CONSTANT
    %import common.WS
    %import common.CNAME -> IDENT
    %ignore WS
''', start='statement', keep_all_tokens=True)


class CalculateTree(Visitor):
    def __init__(self, symbol_table: SymbolTable):
        self.symbol_table = symbol_table
        self.errors = []
        self.last_label = 0
        self.last_for_label = 0

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

    def for_stat(self, tree, attributes):
        if ('for_stat', 'labels') not in attributes:
            start_label = self.last_label
            end_label = self.last_label + 1
            self.last_label += 2
            attributes[('for_stat', 'labels')] = (start_label, end_label)
        else:
            start_label, end_label = attributes[('for_stat', 'labels')]

        attributes[('statement', 'in_loop')] = True
        attributes[('statement', 'loop_end_label')] = end_label
        
        symbol_table = attributes[('for_stat', 'symbol_table')]
        attributes[('statement', 'symbol_table')] = symbol_table

        # generate code
        code = []
        variable = tree.children[2].value
        index = attributes[('atrib_stat', 'index')]
        variable += index

        code.extend(attributes[('atrib_stat', 'code')])
        code.append('{} = {}'.format(variable, code[-1].split()[0]))
        code.append('{}:'.format(start_label))

        code.extend(attributes[('statement', 'code')])

        variable = tree.children[7].value
        index = attributes[('atrib_statd', 'index')]
        variable += index
        code.extend(attributes[('atrib_statd', 'code')])
        code.append('{} = {}'.format(variable, code[-1].split()[0]))

        code.extend(attributes[('expression', 'code')])
        code.append('If {} goto {}'.format(code[-1].split()[0], start_label))
        code.append('{}:'.format(end_label))
        attributes[('for_stat', 'code')] = code


    def if_stat(self, tree, attributes):
        if ('if_stat', 'in_loop') in attributes:
            attributes[('statement', 'in_loop')] = True
            attributes[('else_stat', 'in_loop')] = True

            attributes[('statement', 'loop_end_label')] = attributes[('if_stat', 'loop_end_label')]
            attributes[('else_stat', 'loop_end_label')] = attributes[('if_stat', 'loop_end_label')]

        symbol_table = attributes[('if_stat', 'symbol_table')]
        attributes[('statement', 'symbol_table')] = symbol_table
        attributes[('else_stat', 'symbol_table')] = symbol_table

        condition = attributes[('expression', 'code')]
        statement_code = attributes[('statement', 'code')]
        else_code = attributes[('else_stat', 'code')]

        code = []
        if else_code:
            code.extend(condition)
            else_label = self.last_label
            final_label = self.last_label + 1
            self.last_label += 2
            code.append('If not {} goto {}'.format(condition[-1].split()[0], else_label))

            code.extend(statement_code)
            code.append('goto {}'.format(final_label))

            code.append('{}:'.format(else_label))
            code.extend(else_code)
            code.append('{}:'.format(final_label))
        else:
            code.extend(condition)
            final_label = self.last_label
            self.last_label += 1
            code.append('If not {} goto {}'.format(condition[-1].split()[0], final_label))

            code.extend(statement_code)
            code.append('{}:'.format(final_label))

        attributes[('if_stat', 'code')] = code

    def else_stat(self, tree, attributes):
        if not tree.children:
            attributes[('else_stat', 'code')] = []
            return

        if ('else_stat', 'in_loop') in attributes:
            attributes[('statement', 'in_loop')] = True
            attributes[('statement', 'loop_end_label')] = attributes[('else_stat', 'loop_end_label')]

        symbol_table = attributes[('else_stat', 'symbol_table')]
        attributes[('statement', 'symbol_table')] = symbol_table
        attributes[('else_stat', 'code')] = attributes[('statement', 'code')]

    def statement(self, tree, attributes):
        children = {x.data for x in tree.children if isinstance(x, Tree)}
        tokens = {x.value for x in tree.children if isinstance(x, Token)}
        
        # Check loop
        if ('statement', 'in_loop') in attributes:
            for child in children:
                attributes[(child, 'in_loop')] = True
                attributes[(child, 'loop_end_label')] = attributes[('statement', 'loop_end_label')]
        if 'break' in tokens:
            if ('statement', 'in_loop') not in attributes:
                self.errors.append('Break outside for structure in line {}'.format(tree.children[0].line))

        # Pass on Symbol Table
        symbol_table = attributes[('statement', 'symbol_table')]
        if 'inherit_type' or 'var_or_atrib' in children:
            for child in children:
                attributes[(child, 'symbol_table')] = symbol_table
        else:
            empty_st = deepcopy(symbol_table)
            empty_st.variables = {}
            for child in children:
                attributes[(child, 'symbol_table')] = empty_st

        # generate code
        code = []
        if 'break' in tokens:
            code.append('goto {}'.format(attributes[('statement', 'loop_end_label')]))
        elif ';' in tokens and len(tree.children) == 1:
            pass
        elif 'var_or_atrib' in children:
            variable = tree.children[0].value
            attributes[('var_or_atrib', 'og_type')] = variable
            code = attributes[('var_or_atrib', 'code')]
            if code:
                index = attributes[('var_or_atrib', 'index')]
                variable += index

                code.append('{} = {}'.format(variable, code[-1].split()[0]))
        elif 'dec' and 'inherit_type' not in children:
            child = children.pop()
            code = attributes[(child, 'code')]
        attributes[('statement','code')] = code
        self.code = code

    def print_stat(self, tree, attributes):
        expression_code = attributes[('expression', 'code')]
        variable = expression_code[-1].split()[0]

        code = expression_code
        code.append('out {}'.format(variable))
        attributes[('print_stat', 'code')] = code

    def read_stat(self, tree, attributes):
        variable = tree.children[1].value
        variable += attributes[('lvalue', 'index')]
        code = ['in {}'.format(variable)]
        attributes[('read_stat', 'code')] = code

    def stat_list(self, tree, attributes):
        if not tree.children:
            attributes[('stat_list', 'code')] = []
            return

        if ('stat_list', 'in_loop') in attributes:
            attributes[('statement', 'in_loop')] = True
            attributes[('stat_list\'', 'in_loop')] = True

            attributes[('statement', 'loop_end_label')] = attributes[('stat_list', 'loop_end_label')]
            attributes[('stat_list\'', 'loop_end_label')] = attributes[('stat_list', 'loop_end_label')]

        symbol_table = attributes[('stat_list', 'symbol_table')]
        attributes[('statement', 'symbol_table')] = symbol_table
        attributes[('stat_list\'', 'symbol_table')] = symbol_table

        code = attributes[('statement', 'code')]
        code.extend(attributes[('stat_list\'', 'code')])
        attributes[('stat_list', 'code')] = code

    def return_stat(self, tree, attributes):
        code = attributes[('expressionq', 'code')]
        code.append('goto end')
        attributes[('return_stat', 'code')] = code

    def atrib_stat(self, tree, attributes):
        attributes[('atrib_stat', 'index')] = attributes[('lvalue', 'index')]
        attributes[('atrib_stat', 'code')] = attributes[('expression', 'code')]

    def var_or_atrib(self, tree, attributes):
        if tree.children[0].data == 'var_decl':
            attributes[('var_decl', 'og_type')] = attributes[('var_or_atrib', 'og_type')]

            symbol_table = attributes[('var_or_atrib', 'symbol_table')]
            attributes[('var_decl', 'symbol_table')] = symbol_table

            attributes[('var_decl', 'counter')] = 0
            attributes[('var_or_atrib', 'code')] = []

        else:
            attributes[('var_or_atrib', 'index')] = attributes[('atrib_stat', 'index')]
            attributes[('var_or_atrib', 'code')] = attributes[('atrib_stat', 'code')]

    def atrib_statd(self, tree, attributes):
        attributes[('atrib_statd', 'index')] = attributes[('lvalue', 'index')]
        attributes[('atrib_statd', 'code')] = attributes[('expression', 'code')]

    def expressionq(self, tree, attributes):
        if not tree.children:
            attributes[('expressionq', 'code')] = []
            return

        attributes[('expressionq', 'code')] = attributes[('expression', 'code')]

    def opt_expression(self, tree, attributes):
        if not tree.children:
            return

        attributes[('opt_expression', 'tree')] = attributes[('num_expression', 'tree')]
        attributes[('opt_expression', 'operator')] = tree.children[0].value

    def expression(self, tree, attributes):
        tree = attributes[('num_expression', 'tree')]
        if ('opt_expression', 'tree') in attributes:
            operator = attributes[('opt_expression', 'operator')]
            right_tree = attributes[('opt_expression', 'tree')]
            
            new_tree = EvaluationTree(operator, tree, right_tree)
            tree = new_tree

        attributes[('expression', 'tree')] = tree
        attributes[('expression', 'code')] = tree.three_address_code()

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
            attributes[('unary_vazio', 'val')] = attributes[('unary_vazio', 'left_val')]
            attributes[('unary_vazio', 'type')] = attributes[('unary_vazio', 'left_type')]
            attributes[('unary_vazio', 'tree')] = attributes[('unary_vazio', 'left_tree')] 
        else:
            op = attributes[('mdm_op', 'op')]
            right_val = attributes[('unary_expr','val')]
            right_type = attributes[('unary_expr','type')]
            right_tree = attributes[('unary_expr', 'tree')]
            left_val = attributes[('unary_vazio', 'left_val')] 
            left_type = attributes[('unary_vazio', 'left_type')] 
            left_tree = attributes[('unary_vazio', 'left_tree')]

            root = EvaluationTree(op)           
            root.left = left_tree
            root.right = right_tree

            if (right_type != 'int' or left_type != 'int'):
                print("Type error exception: this expression do not support the types given")
            else:
                if op == "*":
                    result = int(left_val) * int(right_val)
                elif op == "/":
                    result = int(left_val) // int(right_val)
                elif op == "%":
                    result = int(left_val) % int(right_val)
                attributes[('unary_vazio', 'left_val')] = result
                attributes[('unary_vazio', 'left_type')] = left_type
                attributes[('unary_vazio', 'val')] = attributes[('unary_vazio\'', 'val')]
                attributes[('unary_vazio', 'type')] = attributes[('unary_vazio\'','type')]  

            attributes[('unary_vazio', 'left_tree')] = root

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
        #if not tree.children:
        #    pass
        #else:
        #    op = tree.children[0].value
        #    print(op)
        #    print(tree.pretty())
        #    right_val = attributes[('unary_expr','val')]
        #    right_type = attributes[('unary_expr','type')]
        #    left_val = attributes['unary_vazio', 'left_val'] 
        #    left_type = attributes['unary_vazio', 'left_type'] 
        #    if (right_type != 'int' or left_type != 'int'):
        #        print("Type error exception: this expression do not support the types given")
        #    else:
        #        if op == "*":
        #            result = int(left_val) * int(right_val)
        #        elif op == "/":
        #            result = int(left_val) // int(right_val)
        #        elif op == "%":
        #            result = int(left_val) % int(right_val)
        #        attributes[('unary_vazio', 'left_val')] = result
        #        attributes[('unary_vazio', 'left_type')] = left_type
        #        attributes[('unary_vazio', 'val')] = attributes[('unary_vazio\'', 'val')]
        #        attributes[('unary_vazio', 'type')] = attributes[('unary_vazio\'','type')]  

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
            node_name = tokens[0].value
            if ('lvalue', 'index') in attributes:
                node_name += attributes[('lvalue', 'index')]
            attributes[('factor', 'tree')] = EvaluationTree(node_name)
        elif tokens[0] == '(':
            attributes[('factor', 'tree')] = attributes[('num_expression', 'tree')]

    def inherit_type(self, tree, attributes):
        attributes[('var_decl', 'og_type')] = attributes[('ctype', 'type')]

        symbol_table = attributes[('inherit_type', 'symbol_table')]
        attributes[('var_decl', 'symbol_table')] = symbol_table

        attributes[('var_decl', 'counter')] = 0

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

        counter = attributes[('var_decl', 'counter')]
        symbol_table = attributes[('var_decl','symbol_table')]
        if counter == 0:
            self.update_symbol_table(symbol_table, id, attributes[('extra_var','og_type')], attributes[('vector','dimension')], attributes[('vector','n_elements')])
        attributes[('var_decl', 'counter')] += 1

        attributes[('extra_var', 'counter')] = 0
        attributes[('extra_var', 'symbol_table')] = symbol_table

    def extra_var(self, tree, attributes):
        if not tree.children:
            return None

        tokens = [token for token in tree.children if isinstance(token, Token)]
        id = tokens[0].value if len(tokens) == 1 else tokens[1].value
        attributes[('extra_var', 'ident')] = id
        attributes[('extra_var', 'dimension')] = attributes[('vector','dimension')]
        attributes[('extra_var','n_elements')] = attributes[('vector','n_elements')]
        attributes[('extra_var\'','og_type')] = attributes[('extra_var','og_type')]

        counter = attributes[('extra_var', 'counter')]
        symbol_table = attributes[('extra_var','symbol_table')]
        if counter == 0:
            self.update_symbol_table(symbol_table, id, attributes[('extra_var','og_type')], attributes[('vector','dimension')], attributes[('vector','n_elements')])
        attributes[('extra_var', 'counter')] += 1

        attributes[('extra_var\'', 'counter')] = 0
        attributes[('extra_var\'', 'symbol_table')] = symbol_table        

    def vector(self, tree, attrs):
        if not tree.children:
            attrs[('vector', 'dimension')] = 0
            attrs[('vector', 'n_elements')] = 1
            return None
        else:
            size = int(tree.children[1].value)
            attrs[('vector'), ('dimension')] = attrs[('vector\''), ('dimension')] + 1
            attrs[('vector'), ('n_elements')] = attrs[('vector\'','n_elements')] * size

    def update_symbol_table(self, symbol_table, ident, _type, dimension = 0, n_elements = 0):
        if ident in symbol_table.variables:
            self.errors.append('Error: redeclaration of {}'.format(ident))
        
        type = symbol_table.types[_type]
        if dimension == 0:
            symbol_table.variables[ident] = TableEntry(type)
        else:
            array = Array(type, dimension, n_elements*type.required_memory)
            symbol_table.variables[ident] = TableEntry(array)




types = {
    'int': BaseType('int', 4),
    'char': BaseType('char', 1),
    'string': BaseType('string', 0),
    'media': BaseType('media', 32)
    }
symbol_table = SymbolTable(types)
visitor = CalculateTree(symbol_table)

text_path = input()
text_file = open(text_path, 'r')
text = text_file.read()
tree = lark_grammar.parse(text)
try:
    visitor.visit(tree, {('statement', 'symbol_table') : symbol_table})
except Exception as e:
    if visitor.errors:
        for error in visitor.errors:
            print(error)
    else:
        raise e

if visitor.errors:
    for error in visitor.errors:
        print(error)
else:
    print('No errors encountered')

    visitor.code.append('end:')
    for line in visitor.code:
        print(line)
