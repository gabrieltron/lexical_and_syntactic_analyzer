# João Gabriel Trombeta
# Otto Menegasso Pires
# Mathias
# Wagner Braga dos Santos

class Symbol_Table():
	def __init__(self, reserved_words, basics):
		self.reserved_words = reserved_words
		self.basics = basics
		self.symbols_list = []
