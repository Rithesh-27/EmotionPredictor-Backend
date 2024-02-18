# EMTION PREDICTOR USING TENSORFLOW 
# Creating the api for sending prediction responses

import os
import logging
import sys
app_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)
sys.path.append(os.path.join(app_dir,"tfcode"))

from tfcode import prediction
from flask import Flask,request,jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return "Emotion predictor"
@app.route("/predict-emotion",methods=["POST"])
def predict_emotion_api(): 
    data = request.json
    text = data['text']
    predicted_emotion,score = prediction.predict_emotion(text)
    return jsonify({'predicted_emotion': predicted_emotion, 'confidence_percentage':score})

if __name__ == "__main__":
    app.run(debug=True)