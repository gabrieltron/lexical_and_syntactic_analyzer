import sys
import lexical_analyzer as la
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
lexical_analyzer = la.Lexical_Analyzer(source_path, automata_path, symbol_table)

token = ''
errors = []
while token != ('$', ''):
	try:
		token = lexical_analyzer.get_next_token()
	except la.InvalidSymbol as error:
		errors.append(error)
		continue

	print(token)

if errors:
	print('Lexical Analysis not successfull')
	for error in errors:
		print(error)
else:
	print('Symbol table:')
	print(symbol_table.symbols_list)