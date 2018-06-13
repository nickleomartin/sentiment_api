from keras.utils import to_categorical
from models.utils import load_dataset
from models.preprocessing import DocIdTransformer


#X = ['This is a document', 'This is another document!', "And this a third..."]*100
#Y = to_categorical([1,0,1]*100)

Y, X = load_dataset("data/training.txt")


doc_id_transformer = DocIdTransformer()
doc_id_transformer.fit(X)
Y = to_categorical(Y)

# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences

# t = Tokenizer()
# t.fit_on_texts(X)
# vocab_size = len(t.word_index) + 1
# encoded_docs = t.texts_to_sequences(X)
# max_length = 42
# padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')



from models.models import BiLSTM
from models.training import TrainModel

N_W_DIM = len(doc_id_transformer._vocab_builder.vocabulary)
N_CH_DIM = 100
max_sequence_len = doc_id_transformer._vocab_builder._max_sequence_len
model = BiLSTM(n_labels=2, max_seq_len=max_sequence_len, word_vocab_size=N_W_DIM,char_vocab_size=N_CH_DIM)
model.construct()

tm = TrainModel(model=model, preprocessor=doc_id_transformer)
tm.train(X,Y,batch_size=10, epochs=5)


