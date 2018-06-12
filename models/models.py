import keras.backend as K 
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, Lambda, Activation, Reshape
from keras.layers.merge import Concatenate
from keras.models import Model

from models.base_model import BaseModel



class BiLSTM(BaseModel):
	""" 
	Word and Character embeddings + Bidirectional LSTM for sentiment prediction 
	
	Example:
	--------
	from models.models import BiLSTM
	
	model = BiLSTM(n_labels=2,word_vocab_size=1000,char_vocab_size=1000)
	model.construct()

	"""
	def __init__(self, n_labels, word_vocab_size, char_vocab_size=None, word_embedding_dim=100,
					char_embedding_dim=25, word_lstm_size=100, char_lstm_size=25, fc_dim=100, 
					fc_activation='tanh', fc_n_layers=2, dropout=0.5, embeddings=None,
					loss = 'categorical_crossentropy'):
		super(BiLSTM).__init__()
		self._n_labels = n_labels
		self._word_vocab_size = word_vocab_size
		self._char_vocab_size = char_vocab_size
		self._word_embedding_dim = word_embedding_dim
		self._char_embedding_dim = char_embedding_dim
		self._word_lstm_size = word_lstm_size
		self._char_lstm_size = char_lstm_size
		self._fc_dim = fc_dim
		self._fc_activation = fc_activation
		self._fc_n_layers = fc_n_layers
		self._dropout = dropout
		self._embeddings = embeddings
		self._loss = loss
		self.validate_parameters()

	def validate_parameters(self):
		""" TODO: Check parameters are valid """
		if self._char_vocab_size != None:
			if type(self._char_vocab_size) is int:
				self._use_char_embedding = True

	def construct(self):
		""" Build Keras Computational Graph """
		w_ids = Input(batch_shape=(None, None), dtype='int32')
		lengths = Input(batch_shape=(None, None), dtype='int32')

		## Track word, char ids and lengths
		inputs = [w_ids]
		
		## Create embedding layer if not provided
		if self._embeddings is None:
			word_embeddings = Embedding(input_dim=self._word_vocab_size,
										output_dim=self._word_embedding_dim,
										mask_zero=True)(w_ids)

		## If embeddings are provided then load in keras embedding layer
		else:
			word_embeddings = Embedding(input_dim=self._embeddings.shape[0],
										output_dim=self._embeddings.shape[1],
										mask_zero=True,
										weights=[self._embeddings])(w_ids)
		
		## Use character embeddings if specified 
		if self._use_char_embedding:
			ch_ids = Input(batch_shape=(None,None,None), dtype='int32')
			inputs.append(ch_ids)
			char_embeddings = Embedding(input_dim=self._char_vocab_size,
										output_dim=self._char_embedding_dim,
										mask_zero=True)(ch_ids)
			s = K.shape(char_embeddings)
			char_embeddings = Lambda(lambda x: K.reshape(x, shape=(-1, s[-2], self._char_embedding_dim)))(char_embeddings)
			
			## Embed characters with extra BiDirectional layer
			fwd_state = LSTM(self._char_lstm_size, return_state=True)(char_embeddings)[-2]
			bwd_state = LSTM(self._char_lstm_size, return_state=True, go_backwards=True)(char_embeddings)[-2]
			char_embeddings = Lambda(lambda x: K.reshape(x, shape=[-1, s[1], 2*self._char_lstm_size]))(char_embeddings)

			## Concatenate word and char embeddings
			word_embeddings = Concatenate(axis=-1)([word_embeddings, char_embeddings])
		inputs.append(lengths)

		## Build Bidirectional LSTM layer with Fully Connected layers on top
		word_embeddings = Dropout(self._dropout)(word_embeddings)
		h = Bidirectional(LSTM(units=self._word_lstm_size, return_sequences=True))(word_embeddings)
		h = Dropout(self._dropout)(h)
		for fc_layers in range(self._fc_n_layers):
			h = Dense(self._fc_dim, activation=self._fc_activation)(h)
		y_pred = Dense(self._n_labels, activation="softmax")(h)

		## Final model definition with functional API
		self.model = Model(inputs=inputs, outputs=y_pred)

	def __repr__(self):
		""" Ugly printing of class variables """
		return "BiLSTM: %s"%self.__dict__




































