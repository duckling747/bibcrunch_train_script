from pandas.core.frame import DataFrame
from tensorflow.python import keras
from tensorflow.python.keras.backend import concatenate
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers.core import Dropout, Lambda
from tensorflow.python.keras.layers.normalization.batch_normalization import BatchNormalization

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.neighbors import DistanceMetric
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
#nltk.download("wordnet")
#nltk.download("stopwords")

import zipfile
import string
import os

import pandas as pd
import numpy as np



def read_book (zipfile):
    print(zipfile.namelist())
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
#lm = nltk.wordnet.WordNetLemmatizer()
st = nltk.stem.PorterStemmer()
stoppers = stopwords.words("english")

def preprocess_text (text: str):
	text = [ w.lower() for w in text.translate(tbl).split() ]
	return " ".join([ st.stem(w) for w in text if w not in stoppers and w not in ENGLISH_STOP_WORDS ])

vectorizer = TfidfVectorizer(input="content", use_idf=True)

def calc_tfidf (wordlist: list):
    X = vectorizer.fit_transform([wordlist])
    print(X.shape)
    df = pd.DataFrame(X[0].T.todense(), index=vectorizer.get_feature_names(), columns=['TF-IDF'])
    return df

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
			dropout=0.2,
			recurrent_dropout=0.2
		)
	)
	e1 = emb(a)
	e2 = emb(b)
	x1 = lstm(e1)
	x2 = lstm(e2)
	manhattan_distance = lambda x: keras.backend.abs(x[0]-x[1])
	merge = Lambda(function=manhattan_distance, name="manhattan distance", output_shape=lambda x: x[0])([x1, x2])
	#merge = concatenate([x1,x2])
	#merge = BatchNormalization()(merge)
	#merge = Dropout(0.25)(merge)
	#merge = Dense(50, activation="relu")(merge)
	#merge = BatchNormalization()(merge)
	#merge = Dropout(0.25)(merge)
	outputs = Dense(1, activation="sigmoid")(merge)
	model = Model(inputs=[a,b], outputs=outputs)
	model.summary()
	model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])
	return model

def build_dataframe():
	master_df = pd.read_csv("pg_catalog.csv", sep=",", low_memory=False)
	master_df = master_df[(master_df.Language == "en") & (master_df.Type == "Text")]
	master_df.drop(columns=["Bookshelves", "Subjects", "Language", "Type", "Issued"], inplace=True)
	master_df.dropna(how="any", inplace=True)

	# due to time constraints, only take a random sample of books:
	rand_books = master_df.sample(n=100)

	# presume that books are similar if: same LoCC
	similar = [
		master_df["LoCC"].str.contains(book[0])
			for book in rand_books[["LoCC"]].to_numpy()
	]
	df1 = pd.DataFrame(columns=["a", "b", "similar"])
	df1["a"] = rand_books["Text#"]
	df1["b"] = [ master_df[i].sample().iloc[0]["Text#"] for i in similar ]
	df1["similar"] = [int(1) for _ in range(len(df1["a"]))]
	df2 = pd.DataFrame(columns=["a", "b", "similar"])
	df2["a"] = rand_books["Text#"]
	df2["b"] = [ master_df[~i].sample().iloc[0]["Text#"] for i in similar ]
	df2["similar"] = [int(0) for _ in range(len(df2["a"]))]

	df = df1.append(df2, ignore_index=True)
	return df

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

def check_if_exists(a):
	filepath = "gutenberg_books_en/{}.zip"
	found_filepath = (None, None)
	if os.path.exists(filepath.format(str(a))):
		found_filepath = filepath.format(str(a)), "ASCII"
	elif os.path.exists(filepath.format(str(a) + "-0")):
		found_filepath = filepath.format(str(a) + "-0"), "UTF-8"
	elif os.path.exists(filepath.format(str(a) + "-8")):
		found_filepath = filepath.format(str(a) + "-8"), "ISO-8859-1"
	else:
		print ("strike")
	return found_filepath

def load_files_into_memory(df: DataFrame):
	tuples = []
	for a,b,similar in df.itertuples(name=None,index=False):
		a = check_if_exists(a)
		b = check_if_exists(b)
		if a[0] is not None and b[0] is not None:
			a = load_zipfile_to_memory(a[0], a[1])
			b = load_zipfile_to_memory(b[0], b[1])
			if a is not None and b is not None:
				tuples.append((a,b,similar))
	return pd.DataFrame(tuples, columns=["a", "b", "similar"])


df = build_dataframe()
df = load_files_into_memory(df)

print(df)


max_words = 10000
embedding_dim = 100
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")

print("preprocessing texts...")
#df["a"] = df["a"].map(lambda row: preprocess_text(row))
#df["b"] = df["b"].map(lambda row: preprocess_text(row))
df["a"] = [ preprocess_text(row) for row in df["a"] ]
df["b"] = [ preprocess_text(row) for row in df["b"] ]
print("done")

print("fitting tokenizer...")
df["combined"] = df["a"] + " " + df["b"]
tokenizer.fit_on_texts(df["combined"].to_numpy())
print("done")

print("making sequences...")
X1 = make_sequences(tokenizer, books=df["a"].to_numpy(), sequence_max_length=300)
X2 = make_sequences(tokenizer, books=df["b"].to_numpy(), sequence_max_length=300)
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
	input_shape=(300,),
	embedding_matrix=embedding_matrix,
	embedding_dim=embedding_dim,
	lstm_units=256,
	max_words=max_words
)

early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)


X1train, X1test, X2train, X2test, ytrain, ytest = train_test_split(X1, X2, y, test_size=0.2)
X1train, X1val, X2train, X2val, ytrain, yval = train_test_split(X1train, X2train, ytrain, test_size=0.25)

history = model.fit (
	[ X1train, X2train ], ytrain,
	validation_data=([ X1val, X2val ], yval),
	epochs=1000,
	batch_size=64,
	shuffle=True,
	callbacks=[early_stopping]
)

ypred = model.predict([X1test, X2test])
print(ypred)

model.save("model.h5")



