import sys
import lexical_analyzer
import symbol_table

if len(sys.argv) != 3:
	print('Use: python3 compiler.py [source_code_path] [automata_path]')
	sys.exit(0)

reserved_words = {
	'while',
	'do',
	'break',
	'if',
	'then',
	'else',
	'true',
	'false'
}
basics = {
	'int',
	'float',
	'bool'
}

symbol_table = symbol_table.Symbol_Table(reserved_words, basics)

source_path = sys.argv[1]
automata_path = sys.argv[2]
lexical_analyzer = lexical_analyzer.Lexical_Analyzer(source_path, automata_path, symbol_table)

token = ''
while token != ('$', ''):
	token = lexical_analyzer.get_next_token()
	print(token)
print('Symbol table:')
print(symbol_table.symbols_list)