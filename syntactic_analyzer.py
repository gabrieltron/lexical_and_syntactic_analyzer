import grammar
import lexical_analyzer as la

class Syntactic_Analyzer:

    def __init__(self, grammar_path, symbol_table):
        grammar_file = open(grammar_path, 'r')
        grammar_description = grammar_file.read()
        self.grammar = self.create_grammar(grammar_description)

        self.create_first()
        self.create_follow()
        self.create_table()

        self.symbol_table = symbol_table

    def create_grammar(self, _input : str):
        lines = _input.split('\n')
        non_terminals = []
        terminals = set()
        productions = {}

        for line in lines:
            line = line.split('->')
            head = line[0][:-1]
            body = line[1][1:]

            non_terminals.append(head)
            productions[head] = body.split(' | ')

        for body in productions.values():
            for production in body:
                symbols = production.split(' ')
                for symbol in symbols:
                    if symbol not in non_terminals:
                        terminals.add(symbol)

        start = non_terminals[0]
        non_terminals = set(non_terminals)
        return grammar.Grammar(non_terminals, terminals, start, productions)

    def calculate_first(self, symbol, first):
        first[symbol] = set()

        body = self.grammar.p[symbol]
        for production in body:

            production_symbols = production.split()
            episulon = True
            for production_symbol in production_symbols:

                if production_symbol in self.grammar.terminals or production_symbol == '&':
                    first[symbol].add(production_symbol)
                    episulon = False
                    break
                else:
                    if production_symbol not in first:
                        self.calculate_first(production_symbol, first)

                    first[symbol] = first[symbol].union(first[production_symbol])

                    if '&' not in first[production_symbol]:
                        episulon = False
                        break
                    else:
                        first[symbol].remove('&')

            if episulon:
                first[symbol].add('&')


    def create_first(self):
        first = {}
        for symbol in self.grammar.non_terminals:
            if symbol not in first:
                self.calculate_first(symbol, first)
        self.first = first

    def create_follow(self):
        follow = {}

        for non_terminal in self.grammar.non_terminals:
            follow[non_terminal] = set()

        follow[self.grammar.s] = {'$'}

        changed = True
        while changed:
            changed = False
            for non_terminal in self.grammar.non_terminals:

                for production in self.grammar.p[non_terminal]:

                    production_symbols = production.split()
                    for index, symbol in enumerate(production_symbols[:-1]):

                        next_symbol = production_symbols[index+1]
                        if symbol in self.grammar.terminals:
                            continue

                        elif next_symbol in self.grammar.terminals:
                            if next_symbol not in follow[symbol]:
                                follow[symbol].add(next_symbol)
                                changed = True

                        else:
                            for f in self.first[next_symbol]:
                                if f not in follow[symbol] and f != '&':

                                    follow[symbol].add(f)
                                    changed = True

                for production in self.grammar.p[non_terminal]:
                    head = non_terminal
                    r_production_symbols = list(reversed(production.split()))

                    for symbol in r_production_symbols:
                        if symbol in self.grammar.terminals:
                            break
                        else:
                            for f in follow[head]:
                                if f not in follow[symbol] and f not in self.first[symbol]:
                                    follow[symbol].add(f)
                                    changed = True
                        if '&' not in self.first[symbol]:
                            break

                    for i,symbol in enumerate(r_production_symbols[:-1]):
                        prev_symbol = r_production_symbols[i+1]

                        if symbol in self.grammar.terminals or prev_symbol in self.grammar.terminals:
                            continue
                        elif '&' in self.first[symbol]:
                            for f in follow[symbol]:
                                if f not in follow[prev_symbol]:
                                    follow[prev_symbol].add(f)
                                    changed = True 

        self.follow = follow

    def is_terminal_origin(self, symbol, origin_production):
        for production in origin_production:
            if production in self.grammar.terminals:
                if production == symbol:
                    return True
                else:
                    return False
            elif symbol in self.first[production]:
                return True
            else:
                if '&' not in self.first[production]:
                    break

        return False

    def create_table(self):
        table = {}

        for non_terminal in self.grammar.non_terminals:
            table[non_terminal] = {}

            for symbol in self.first[non_terminal]:
                if symbol != '&':

                    origin = ''
                    for production in self.grammar.p[non_terminal]:

                        production = production.split()
                        terminal_origin = self.is_terminal_origin(symbol, production)

                        if terminal_origin:
                            origin = ' '.join(production)
                            break

                    if origin:
                        table[non_terminal][symbol] = origin
                else:
                    for follow_symbol in self.follow[non_terminal]:
                        table[non_terminal][follow_symbol] = '&'

            #for body in self.grammar.p[non_terminal]:
            #    symbols = body.split()
            #    for symbol in symbols:
            #        if symbol in self.grammar.terminals and symbol not in table[non_terminal]:
            #            table[non_terminal][symbol] = body

            if '&' not in self.first[non_terminal]:
                for follow_symbol in self.follow[non_terminal]:
                    if follow_symbol not in table[non_terminal]:
                        table[non_terminal][follow_symbol] = 'sync'


        self.table = table

    def add_error(self, errors, symbol, line):
        if symbol == '$':
            error = InvalidSyntax("Unexpected EOF while parsing.")
        else:
            error = InvalidSyntax("Unexpected {} while parsing in line {}.".format(symbol, line))
        errors.append(error)

    def parse_code(self, alex):
        stack = ['$']
        stack.append(self.grammar.s)
        word = []
        errors = []

        while stack:
            if not word:
                try:
                    token = alex.get_next_token()
                except la.InvalidSymbol as error:
                    errors.append(error)
                    continue
                terminal = token[0]
                word.append(terminal)

                
            compare = stack.pop()

            #print(terminal)
            #print(compare)
            #if compare in self.table:
            #    print(self.table[compare])
            #print('---------')
            
            if compare == terminal:
                word = word[1:]
            elif compare == '$':
                self.add_error(errors, terminal, alex.lines_analyzed)
                word = word[:1]

            elif compare in self.grammar.terminals:
                self.add_error(errors, terminal, alex.lines_analyzed)

            else:
                if terminal in self.table[compare]:
                    transition = self.table[compare][terminal]
                    if transition != 'sync':
                        transition = list(reversed(transition.split()))
    
                        if transition != ['&']:
                            for symbol in transition:
                                stack.append(symbol)
                    else:
                        self.add_error(errors, terminal, alex.lines_analyzed)
                        if len(stack) == 1:
                            word = word[1:]
                            stack.append(compare)
                elif terminal == '$':
                    self.add_error(errors, terminal, alex.lines_analyzed)
                    break

                else:
                    self.add_error(errors, terminal, alex.lines_analyzed)
                    word = word[1:]
                    stack.append(compare)

        for error in errors:
            print(error)
        if errors:
            return False
        else:
            return True


class InvalidSyntax(Exception):
    pass

