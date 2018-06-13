from keras.utils import to_categorical
from models.utils import load_dataset
from models.preprocessing import DocIdTransformer
from models.models import BiLSTM
from models.training import TrainModel

## Dataset
X, Y = load_dataset("data/training.txt")

## Index and transform to sequences
doc_id_transformer = DocIdTransformer()
doc_id_transformer.fit(X)

## Convert list of class labels to categorical
Y = to_categorical(Y)

## Train model
N_W_DIM = len(doc_id_transformer._vocab_builder.vocabulary)
max_word_seq_len = doc_id_transformer._vocab_builder._max_word_sequence_len

model = BiLSTM(n_labels=2, max_word_seq_len=max_word_seq_len, word_vocab_size=N_W_DIM)
model.construct()

tm = TrainModel(model=model, preprocessor=doc_id_transformer)
tm.train(X, Y, batch_size=64, epochs=100)


