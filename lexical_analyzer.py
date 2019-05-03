import automata

class Lexical_Analyzer():

    def __init__(self, source_path, automata_path, symbol_table):
        self.symbol_table = symbol_table

        automata_file = open(automata_path, 'r')
        automata_description = automata_file.read()
        self.automata = self.create_automata(automata_description)

        source_file = open(source_path, 'r')
        self.source_code = source_file.read()

        self.lines_analyzed = 1

    def create_automata(self, content):
        content = content.split('\n')
        states = content[0].split()
        alphabet = content[1].split()
        initial_state = content[2]
        final_states = content[3].split()

        transitions = {}
        for state in states:
            transitions[state] = {}
            for symbol in alphabet:
                transitions[state][symbol] = []
            transitions[state]['&'] = []
        for transition in content[4:]:
            t = transition.split()
            source = t[0]
            destiny = t[2]
            symbol = t[3]
            transitions[source][symbol].append(destiny)

        _automata = automata.Automata(states, alphabet, initial_state, final_states, transitions)
        return _automata

    def ignore_blanks(self):
        while self.source_code and (
                self.source_code[0] == ' ' or
                self.source_code[0] == '\n' or
                self.source_code[0] == '\t'
            ): 
            if self.source_code[0] == '\n':
                self.lines_analyzed += 1
            self.source_code = self.source_code[1:]

    def get_next_token(self):        
        if len(self.source_code) == 0:
            return ('$', '')
        
        self.ignore_blanks()

        if len(self.source_code) == 0:
            return ('$', '')

        state, is_valid, symbols_read = self.automata.run(self.source_code)
        symbol_read = self.source_code[:symbols_read+1]

        if not is_valid:
            self.source_code = self.source_code[symbols_read+1:]
            raise InvalidSymbol("Symbol {} in line {} is not present in the grammar.".format(symbol_read, self.lines_analyzed))

        if state == 'id':
            if symbol_read in self.symbol_table.reserved_words:
                token = (symbol_read, '')
            elif symbol_read in self.symbol_table.basics:
                token = (symbol_read, '')
            else:
                token = ('ident', symbol_read)
                self.symbol_table.symbols_list.append(token)

        elif state == 'int':
            token = ('int-const', symbol_read)
            self.symbol_table.symbols_list.append(token)

        elif state == 'string':
            token = ('string-const', symbol_read)
            self.symbol_table.symbols_list.append(token)

        else:
            token = (symbol_read, '')

        self.source_code = self.source_code[symbols_read+1:]
        return token

class InvalidSymbol(Exception):
    pass