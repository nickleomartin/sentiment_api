from collections import Counter

class DatasetWrapper(object):
	""" 
	Converts vocabulary to ids
	
	Example:
	--------
	from models.dataset_wrapper import DatasetWrapper
	
	documents = ["This is a sentence.", "This is another sentence!"]

	dw = DatasetWrapper()
	dw.add_documents(documents)
	dw.build()
	"""
	def __init__(self, max_vocab_size=10000, lowercase=True, unk_token=True, specials=('<pad>',)):
		self._max_vocab_size = max_vocab_size
		self._lowercase = lowercase
		self._unk_token = unk_token
		self._token2id = {token: i for i, token in enumerate(specials)}
		self._id2token = list(specials)
		self._token_count = Counter()

	def __len__(self):
		return len(self._token2id)

	def process_token(self, token):
		""" Simple lowercasing """
		if self._lowercase:
			token = token.lower()
		return token

	def add_token(self, token):
		""" Update token_count with tokem """
		token = self.process_token(token)
		self._token_count.update([token])

	def add_documents(self, documents):
		""" Loop through documents, process and append to token_count """
		for document in documents:
			document = map(self.process_token, document)
			self._token_count.update(document)

	def doc_to_id(self, document):
		""" Convert document to id """
		document = map(self.process_token, document)
		return [self.token_to_id(token) for token in document]

	def id_to_doc(self, ids):
		""" Convert set of ids to documents """ 
		return [self.id_to_token(idx) for idx in ids]

	def token_to_id(self, token):
		""" Retrieve id given token """
		token = self.process_token(token)
		return self._token2id.get(token, len(self._token2id) - 1)

	def id_to_token(self, idx):
		""" Retrieve token given id """
		return self._id2token[idx]

	def build(self):
		""" Build the vocabulary """
		token_freq = self._token_count.most_common(self._max_vocab_size)
		idx = len(self.vocab)
		for token, _ in token_freq:
			self._token2id[token] = idx
			self._id2token.append(token)
			idx += 1
		
		## Handle unknown words
		if self._unk_token:
			unk = '<unk>'
			self._token2id[unk] = idx
			self._id2token.append(unk)

	@property
	def vocab(self):
		return self._token2id

	@property
	def reverse_vocab(self):
		return self._id2token



