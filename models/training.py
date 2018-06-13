import numpy as np



class TrainModel(object):
	"""
	Train model using Keras fit_generator method

	Example:
	--------
	from models.models import BiLSTM
	from models.training import TrainModel
	
	## Initilze model

	## Pass to training class

	## Save model weights and params

	"""
	def __init__(self, model, preprocessor=None):
		self._model = model
		self._preprocessor = preprocessor

	def data_generator(self, X, Y, batch_size, shuffle, preprocessor, batches_per_epoch):
		""" Return batch generator of X and y """ 
		data_size = len(X)
		while True:
			indices = np.arange(data_size)

			## Randomly permute X indices
			if shuffle:
				indices = np.random.permutation(indices)

			## Loop through batch, index X and Y and preprocess
			for batch_num in range(batches_per_epoch):
				start_index = batch_num * batch_size
				end_index = min((batch_num + 1) * batch_size, data_size)
				x_batch = [X[j] for j in indices[start_index:end_index]]
				y_batch = [Y[j] for j in indices[start_index:end_index]]
				yield preprocessor.transform(x_batch, y_batch)

	def batch_iterator(self, X, Y, batch_size=1, shuffle=True, preprocessor=None):
		""" Return batch iterator """
		batches_per_epoch = int(len(X) - 1)/ batch_size + 1
		X_generator = self.data_generator(X, Y, batch_size, shuffle, preprocessor, batches_per_epoch)
		return batches_per_epoch, X_generator

	def train(self, x_train, y_train, epochs=1, batch_size=64, verbose=1,
				callbacks=None, shuffle=True):
		""" Create batch generator and train the model """
		## Get training generator
		training_data_steps, training_data_generator = batch_iterator(x_train, y_train, )

		## Train the model
		self._model.fit_generator(generator=training_data_generator,
									steps_per_epoch=training_data_steps,
									epochs=epochs,
									callbacks=callbacks,
									verbose=verbose)












































