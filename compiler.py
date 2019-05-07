import sys
import lexical_analyzer as la
import symbol_table
import syntactic_analyzer as sa
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

source_path = sys.argv[1]
automata_path = sys.argv[2]
lexical_analyzer = la.Lexical_Analyzer(source_path, automata_path, symbol_table)

grammar_path = sys.argv[3]
syntactic_analyzer = sa.Syntactic_Analyzer(grammar_path, symbol_table)
valid_word = syntactic_analyzer.parse_code(lexical_analyzer)

if valid_word:
	print('Code is valid')
else:
	print('Code is invalid')


#token = ''
#errors = []
#while token != ('$', ''):
#	try:
#		token = lexical_analyzer.get_next_token()
#	except la.InvalidSymbol as error:
#		errors.append(error)
#		continue
#
#	print(token)
#
#if errors:
#	print('Lexical Analysis not successfull')
#	for error in errors:
#		print(error)
#else:
#	print('Symbol table:')
#	print(symbol_table.symbols_list)