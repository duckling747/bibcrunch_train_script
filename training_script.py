from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers.core import Lambda

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.backend import abs as kabs

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords
#nltk.download("wordnet")
#nltk.download("stopwords")

import zipfile
import string
import os
import random
from multiprocessing import Pool
import pickle

import pandas as pd
import numpy as np


#os.environ["TOKENIZERS_PARALLELISM"] = "true"

def read_book (zipfile):
    return zipfile.read(zipfile.namelist()[0])

def load_zipfile_to_memory (fp: str, encoding: str):
	book_as_string = None
	with zipfile.ZipFile(fp, "r") as zfile:
		try:
			book_as_string = read_book(zfile).decode(encoding)
		except UnicodeDecodeError:
			pass
	return book_as_string


tbl = str.maketrans("","",string.punctuation+string.digits)
lm = nltk.wordnet.WordNetLemmatizer()
stoppers = stopwords.words("english")

def preprocess_text (text: str):
	text = [ w.lower() for w in text.translate(tbl).split() ]
	return " ".join([ lm.lemmatize(w) for w in text if w not in stoppers and w not in ENGLISH_STOP_WORDS ])

vectorizer = TfidfVectorizer(input="content", use_idf=True)

def get_dfidf_similarity_estimate (book_pair: tuple):
    X = vectorizer.fit_transform(book_pair)
    X = cosine_similarity(X)[0][1]
    return X

def build_siamese_network (input_shape, embedding_matrix, embedding_dim, max_words, lstm_units):
	a = Input(shape=input_shape)
	b = Input(shape=input_shape)
	emb = Embedding(
		max_words,
		embedding_dim,
		input_length=input_shape[0],
		weights=[embedding_matrix],
		trainable=False
	)
	lstm = Bidirectional(LSTM(
		units=lstm_units,
		return_sequences=False,
		dropout=0.3,
		recurrent_dropout=0.3,
	))
	x1 = emb(a)
	x2 = emb(b)
	x1 = lstm(x1)
	x2 = lstm(x2)
	manhattan = lambda x: kabs(x[0]-x[1])
	merge = Lambda(function=manhattan, output_shape=lambda x: x[0], name="manhattan")([x1,x2])
	outputs = Dense(1, activation="sigmoid")(merge)
	model = Model(inputs=[a,b], outputs=outputs)
	model.summary()
	model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
	return model

def get_embeddings_index():
	embeddings_index = {}
	with open("glove.6B.100d.txt", "r") as f:
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype="float32")
			embeddings_index[word] = coefs
	return embeddings_index

def make_sequences (tokenizer: Tokenizer, books: np.ndarray, sequence_max_length: int):
	books = tokenizer.texts_to_sequences(books)
	books = pad_sequences(books,
		maxlen=sequence_max_length,
		padding="post")
	return books

def get_embedding_matrix(tokenizer: Tokenizer, max_words: int, embedding_dim: int):
	embeddings_index = get_embeddings_index()
	embedding_matrix = np.zeros((max_words, embedding_dim))
	for word, i in tokenizer.word_index.items():
		if i < max_words:
			embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None and i < max_words:
			embedding_matrix[i] = embedding_vector
	return embedding_matrix


def get_text (path: str):
	if path.endswith("-8.zip"):
		return load_zipfile_to_memory(path, "ISO-8859-1")
	elif path.endswith("-0.zip"):
		return load_zipfile_to_memory(path, "UTF-8")
	elif path.endswith(".zip"):
		return load_zipfile_to_memory(path, "ASCII")
	else:
		return None

def handle_stuff(t: tuple):
	fa, fb, dir = t
	print (f"processing {fa} & {fb}...")
	fa = get_text(os.path.join(dir, fa))
	fb = get_text(os.path.join(dir, fb))
	if fa is not None and fb is not None:
		fa = preprocess_text(fa)
		fb = preprocess_text(fb)
		sim = get_dfidf_similarity_estimate (( fa, fb ))
		ida = t[0].replace("-8.zip","").replace("-0.zip","").replace(".zip", "")
		idb = t[1].replace("-8.zip","").replace("-0.zip","").replace(".zip", "")
		return {"id_a": ida, "id_b": idb, "a": fa, "b": fb, "similar": sim}
	return None

def load_random_files_preprocess_and_calculate_diff(amount_of_pairs: int):
	dir = "gutenberg_books_en/"
	filenames_a = random.sample(os.listdir(dir), amount_of_pairs)
	filenames_b = random.sample(os.listdir(dir), amount_of_pairs)
	nltk.corpus.wordnet.ensure_loaded()
	frames = None
	with Pool(processes=7) as p:
		frames = p.map(handle_stuff, [(t[0],t[1],dir) for t in zip(filenames_a, filenames_b)])
	return [f for f in frames if f is not None]

def sample_and_trim(df, sample_size):
	similar = df[df["similar"] >= 0.40].sample(sample_size)
	not_similar = df[df["similar"] <= 0.15].sample(sample_size)
	del df
	similar["similar"] = int(1)
	not_similar["similar"] = int(0)
	return pd.concat([similar, not_similar], ignore_index=True)


df = None
saved_preprocessed = "preprocessed_data.pkl"
if os.path.exists(saved_preprocessed):
	df = pd.read_pickle(saved_preprocessed)
else:
	print ("preprocessing texts...")
	frames = load_random_files_preprocess_and_calculate_diff(amount_of_pairs=400)
	print ("done")
	print ("saving to disk...")
	pd.DataFrame(frames).to_pickle(saved_preprocessed)
	print ("done")
	print ("please rerun the script to continue")
	exit(0)

df = sample_and_trim(df, sample_size=50)
print(df)

max_words = 10000 # i.e. unique words
embedding_dim = 100 # see pretrained Glove file
sequence_max_length = 4000
lstm_units_top_layer = 64
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")


print("fitting tokenizer...")
df["combined"] = df["a"] + " " + df["b"]
tokenizer.fit_on_texts(df["combined"].to_numpy())
df.drop(columns=["combined"])
print("done")

print("making sequences...")
X1 = make_sequences(tokenizer, books=df["a"].to_numpy(), sequence_max_length=sequence_max_length)
X2 = make_sequences(tokenizer, books=df["b"].to_numpy(), sequence_max_length=sequence_max_length)
print("done")
y = df["similar"].to_numpy()
del df
#print(X1)
for i in X1:
	print (i)
#print(X2)
for i in X2:
	print (i)
print("len X1:", len(X1))
print("len X2:", len(X2))
print (y)
print("len y:", len(y))


embedding_matrix = get_embedding_matrix(tokenizer=tokenizer, max_words=max_words, embedding_dim=embedding_dim)

model = build_siamese_network (
	input_shape=(sequence_max_length,),
	embedding_matrix=embedding_matrix,
	embedding_dim=embedding_dim,
	lstm_units=lstm_units_top_layer,
	max_words=max_words
)

early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)


X1train, X1test, X2train, X2test, ytrain, ytest = train_test_split(X1, X2, y, test_size=0.2)
X1train, X1val, X2train, X2val, ytrain, yval = train_test_split(X1train, X2train, ytrain, test_size=0.25)

history = model.fit (
	[ X1train, X2train ], ytrain,
	validation_data=([ X1val, X2val ], yval),
	epochs=1000,
	batch_size=8,
	shuffle=True,
	callbacks=[early_stopping]
)

ypred = model.predict([X1test, X2test])
print(ypred)

model.save("model.h5")

with open("tokenizer.pkl", "wb") as h:
	pickle.dump(tokenizer, h, protocol=pickle.HIGHEST_PROTOCOL)

