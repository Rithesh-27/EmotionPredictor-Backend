# EMOTION PREDICTION MODEL USING TENSORFLOW
# Loading the model and testing/making predictions on raw strings

# necessary imports
import contractions
import pickle
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")


import tensorflow as tf
import numpy as np

# loading the model
model = tf.keras.models.load_model("C:/Users/rithe/Desktop/python/machine_learning/projects/emotion_predictor/model/emotion_predictor.keras")

# importing training data for creating tokenizer
with open("C:/Users/rithe/Desktop/python/machine_learning/projects/emotion_predictor/data/tokenizer.pickle","rb") as handle:
    tokenizer = pickle.load(handle)
    
# function to preprocess text
def preprocess_text_list(text_list):
    for i in range(len(text_list)):
        text_list[i] = text_list[i].lower()
        text_list[i] = contractions.fix(text_list[i])

    sequences = tokenizer.texts_to_sequences(text_list)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)
    return padded_sequences

# example raw string
raw_text_list = ["i am happy and excited about this prject", "i was scared to death"]
processed_text_list = preprocess_text_list(raw_text_list)
predictions = model.predict(processed_text_list)
print(predictions)

def decode_labels(labels):
    decode_table = {0:'joy',1:'anger',2:'love',3:'sadness',4:'fear',5:'surprise'}
    ret = []
    for label in labels:
        ret.append(decode_table[label])
    return ret
    

predicted_labels = np.argmax(predictions, axis=1)
sentiments  = decode_labels(predicted_labels)


print("Predicted sentiments:", sentiments)
