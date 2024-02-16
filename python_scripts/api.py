# EMTION PREDICTOR USING TENSORFLOW 
# Creating the api for sending prediction responses

import contractions
import pickle
import os
import logging
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
import numpy as np
from flask import Flask,request,jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# function to predict the sentiment for given text
def predict_emotion(text):
    # loading the model
    model = tf.keras.models.load_model("C:/Users/rithe/Desktop/python/machine_learning/projects/emotion_predictor/model/emotion_predictor.keras")

    # importing training data for creating tokenizer
    with open("C:/Users/rithe/Desktop/python/machine_learning/projects/emotion_predictor/data/tokenizer.pickle","rb") as handle:
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

@app.route("/")
def index():
    return "Emotion predictor"
@app.route("/predict-emotion",methods=["POST"])
def predict_emotion_api(): 
    data = request.json
    text = data['text']
    predicted_emotion,score = predict_emotion(text)
    return jsonify({'predicted_emotion': predicted_emotion, 'confidence_percentage':score})

if __name__ == "__main__":
    app.run(debug=True)