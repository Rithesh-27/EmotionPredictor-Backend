# EMOTION PREDICTOR MODEL USING TENSORFLOW
# Creating and training the model

# accessing CUDA toolkit for gpu accelerartion
import contractions
import pickle
import os
program_files_path = os.environ.get("ProgramFiles")
os.add_dll_directory(os.path.join(program_files_path,"NVIDIA GPU Computing Toolkit","CUDA","v11.2","bin"))

# importing all the required libraries
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from keras import layers,losses


# DATA LOADING AND PREPROCESSING

# loadind text data into pandas dataframe
app_dir = os.path.dirname(os.path.abspath(__file__))
train_file = os.path.join(app_dir,"data","train.txt")
test_file = os.path.join(app_dir,"data","test.txt")
val_file = os.path.join(app_dir,"data","val.txt")
train_df = pd.read_table(train_file,delimiter=";",header=None)
val_df = pd.read_table(val_file,delimiter=";",header=None)
test_df = pd.read_table(test_file,delimiter=";",header=None)

raw_data = pd.concat([train_df,val_df,test_df])
raw_data.columns = ["content","sentiment"]

# function to convert text into lowercase and remove contractions
def clean_text(text):
    text = text.lower()
    text = contractions.fix(text)
    return text

raw_data["content"] = raw_data["content"].apply(clean_text)

# creating and assigning integer encoding for emotions
encode_table = {'joy':0,'anger':1,'love':2,'sadness':3,'fear':4,'surprise':5}

def encode_labels(text):
    return encode_table[text]

raw_data["sentiment"] = raw_data["sentiment"].apply(encode_labels)

# splitting raw_data into training and testing data
data_len = len(raw_data["content"])
train_df = raw_data[:int(data_len * 0.75)].copy()
test_df = raw_data[int(data_len * 0.75):].copy()

# splitting into features and labels for training and testing values
train_features = train_df["content"].tolist()
train_labels = train_df["sentiment"].values

test_features = test_df["content"].tolist()
test_labels = test_df["sentiment"].values


# MODEL CREATION

# using GloVe to assign tokens for feeding into model
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
GLOVE_PATH = os.path.join(app_dir,"GloVe_files","glove.6B.100d.txt")


# function to load the GloVe embedding layer form path
def load_glove_embeddings(path):
    embeddings_index = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

glove_embeddings_index = load_glove_embeddings(GLOVE_PATH)

# using tokenizer and padding to clean up text data
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(train_features)
sequences_train = tokenizer.texts_to_sequences(train_features)
sequences_test = tokenizer.texts_to_sequences(test_features)
train_features = keras.preprocessing.sequence.pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
test_features = keras.preprocessing.sequence.pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

# saving the tokenizer ogject for future usage in prediction
with open(os.path.join(app_dir,"data","tokenizer.pickle"),"wb") as handle:
    pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)

# creating embedding matrix with words present in GloVe vocabulary
word_index = tokenizer.word_index
num_words = min(len(word_index) + 1, len(glove_embeddings_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= num_words:
        continue
    embedding_vector = glove_embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# creating the model
model = keras.Sequential([
    layers.Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM, embeddings_initializer=keras.initializers.Constant(embedding_matrix), input_length=MAX_SEQUENCE_LENGTH, trainable=False),
    layers.LSTM(units=128,dropout=0.2,recurrent_dropout=0.2),
    layers.Dense(units=6,activation="softmax")
])

# compiling the model
model.compile(loss=losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

# training the model
with tf.device("/GPU:0"):
    model.fit(train_features, train_labels, batch_size=128, epochs=10, validation_split=0.1)

# evaluating the model on test data
model.evaluate(test_features,test_labels)

# saving the model
model.save(os.path.join(app_dir,"model","emotion_predictor.keras"))