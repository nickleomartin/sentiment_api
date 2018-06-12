


def load_dataset(file_path):
	""" 
	Load training dataset 

	Example:
	--------
	from models.utils import load_dataset

	labels, sentences = load_dataset("data/training.txt")
	"""
	sentences, labels = [], [] 
	words, tags = [], []
	with open(file_path) as f:
		for i,line in enumerate(f):
			## Remove any trailing characters 
			line = line.rstrip()

			## Split on tab
			label, sentence = line.split("\t")
			
			## Store and return
			sentences.append(sentence)
			labels.append(label)

	return (labels, sentences)			


## TODO: Consider class to wrap embeddings and handle logic
def filter_embeddings(embeddings, vocab, dim):
	if not isinstance(embeddings, dict):
		return
	_embeddings = np.zeros([len(vocab), dim])
	for word in vocab:
		if word in embeddings:
			word_idx = vocab[word]
			_embeddings[word_idx] = embeddings[word]

	return _embeddings







