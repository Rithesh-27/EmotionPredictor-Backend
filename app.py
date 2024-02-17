# EMTION PREDICTOR USING TENSORFLOW 
# Creating the api for sending prediction responses

import contractions
import pickle
import os
import logging
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
    app_dir = os.path.dirname(os.path.abspath(__file__))
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