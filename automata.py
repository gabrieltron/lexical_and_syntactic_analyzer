#coding: utf-8
class Automata():
    def __init__(self, states, alphabet, start_state, final_states, transitions):
        self.states = states
        self.alphabet = alphabet
        self.start_state = start_state
        self.final_states = final_states
        self.transitions = transitions

    def __str__(self):
        string = ' '.join(self.states) + '\n'
        string += ' '.join(self.alphabet) + '\n'
        string += self.start_state + '\n'
        string += ' '.join(self.final_states) + '\n'
        for source, symbol_dict in self.transitions.items():
            for symbol, destiny_list in symbol_dict.items():
                for destiny in destiny_list:
                    string += source + ' -> ' + destiny + ' ' + symbol + '\n'

        return string

    def run(self, string):
        current_state = self.start_state
        last_read = 0
        for index, symbol in enumerate(string):
            # check if it's an exception state, that is,
            # stay on the state unless seen a ginve symbol
            if current_state[0] != '*':
                # it's not, check if transition exists
                if symbol not in self.alphabet:
                    break

                elif not self.transitions[current_state][symbol]:
                    break

                next_state = self.transitions[current_state][symbol][0]
            else:
                # it is, check if should go to another state or stay
                if (symbol in self.transitions[current_state] and
                        self.transitions[current_state][symbol]):
                    next_state = self.transitions[current_state][symbol][0]
                else:
                    next_state = current_state

            last_read = index
            current_state = next_state
        
        is_final = current_state in self.final_states
        return (current_state, is_final,last_read)
