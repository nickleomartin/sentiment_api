





class Vocabulary(object):
	""" Maps tokens to ints """

	def __init__(self, max_vocab_size=None, lowercase=True, unk_token=True, specials=('<pad>',)):
		self._max_vocab_size = max_vocab_size
		self._lowercase = lowercase
		self._unk_token = unk_token
		self._token2id = {token: i for i, token in enumerate(specials)}
		self._id2token = list(specials)
		self._token_count = Counter()

	def __len__(self):
		return len(self._token2id)

	def add_token(self, token):
		token = self.process_token(token)
		self._token_count.update([token])
		

































































