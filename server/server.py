from flask import Flask, request, jsonify
from flask_cors import CORS
from main import predict, preprocess
from main import ConvolutionalNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
import proselint
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        title, text, subject, date = data["title"], data["content"], data["category"], data["date"]
        print(title, text, subject, date)
        test_input = preprocess(title, text, subject, date)
        print(test_input)
        is_fake = predict(test_input)
        print(is_fake)

        return jsonify({"isFake": 1-is_fake})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
