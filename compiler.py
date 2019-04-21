import sys
import lexical_analyzer as la
import symbol_table
#import syntactic_analyzer as sa
from pprint import pprint

reserved_words = {
	'class',
	'extends',
	'constructor',
	'break',
	'print',
	'read',
	'return',
	'super',
	'if',
	'else',
	'for',
	'new',
	'null'
}
basics = {
	'int',
	'string'
}

symbol_table = symbol_table.Symbol_Table(reserved_words, basics)

automata_path = sys.argv[1]
source_path = input("Type file path\n")
lexical_analyzer = la.Lexical_Analyzer(source_path, automata_path, symbol_table)

#source_path = sys.argv[1]
#grammar_path = sys.argv[3]
#syntactic_analyzer = sa.Syntactic_Analyzer(grammar_path, symbol_table)

#valid_word = syntactic_analyzer.parse_code(lexical_analyzer)
#
#if valid_word:
#	print('Word is valid')
#else:
#	print('Word is invalid')


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