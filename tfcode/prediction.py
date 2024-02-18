# EMOTION PREDICTION MODEL USING TENSORFLOW
# Loading the model and testing/making predictions on raw strings

# necessary imports
import contractions
import pickle
import os

import tensorflow as tf
import numpy as np
app_dir = os.getcwd()

# function to predict the sentiment for given text
def predict_emotion(text):
    # loading the model
    model_file = os.path.join(app_dir,"model","emotion_predictor.keras")
    model = tf.keras.models.load_model(model_file)

    # importing training data for creating tokenizer
    pickle_file = os.path.join(app_dir,"data","tokenizer.pickle")
    with open(pickle_file,"rb") as handle:
        tokenizer = pickle.load(handle)
        
    # function to preprocess text
    def preprocess_text(text):
        text = text.lower()
        text = contractions.fix(text)
        sequences = tokenizer.texts_to_sequences([text])
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)
        return padded_sequences
    
    processed_text = preprocess_text(text)
    predictions = model.predict(processed_text)

    # function to decode given label
    def decode_labels(label):
        decode_table = {0:'joy',1:'anger',2:'love',3:'sadness',4:'fear',5:'surprise'}
        return decode_table[label]
        

    predicted_label = np.argmax(predictions, axis=1)[0]
    sentiment  = decode_labels(predicted_label)
    prediction_score = np.max(predictions) * 100

    return sentiment,prediction_score
