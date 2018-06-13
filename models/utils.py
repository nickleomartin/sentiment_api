import numpy as np


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
def filter_embeddings(embeddings, vocab, embedding_dim):
	"""
	Filter word embeddings by vocab   

	Example:
	--------
	import numpy as np
	from models.utils import filter_embeddings
	
	embeddings = {"this": np.array([0]*100), "vocab": np.array([0]*100)} 
	vocab = {"this":0, "is":1, "vocab":2}

	word_embeddings = filter_embeddings(embeddings, vocab, embedding_dim=100)
	"""
	## TODO: Add useful error messages
	if not isinstance(embeddings, dict):
		return
	
	## TODO: Check embedding_dim == embeddings.shape[0]

	## Loop through vocab and obtain embeddings
	_embeddings = np.zeros([len(vocab), embedding_dim])
	for word in vocab:
		if word in embeddings:
			word_idx = vocab[word]
			_embeddings[word_idx] = embeddings[word]
	return _embeddings


## TODO: Add function to load Google vectors 

def load_glove(file_path):
	""" Loads Glove vectors into np.array """
	word_vect_dict = {}
	with open(file_path) as f:
		for line in f:
			line = line.split(' ')
			word = line[0]
			gl_vector = np.array([float(val) for val in line[1:]])
			word_vect_dict[word] = gl_vector
	return word_vect_dict




