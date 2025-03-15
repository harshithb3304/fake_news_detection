import torch
import torch.nn as nn
import torch.nn.functional as F
import proselint
import nltk
from nltk.corpus import stopwords

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()

        # More dense 1D Convolutional layers
        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),  # (64, 17)
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2),  # (64, 8)

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),  # (128, 8)
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2),  # (128, 4)

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),  # (256, 4)
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2),  # (256, 2)

            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),  # (512, 2)
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2),  # (512, 1)

            nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=1),  # (1024, 1)
            nn.Tanh(),
        )

        # Fully connected layers after the convolutional layers
        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512),  # Flattened input to the fully connected layer (1024 channels)
            nn.Tanh(),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 2),  # Output layer for binary classification
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, 17)  # Reshape input to [batch_size, 1, 17]
        x = self.conv_layer(x)  # Apply convolutional layers
        x = torch.flatten(x, 1)  # Flatten the tensor before feeding it into the fully connected layers
        x = self.fc_layer(x)  # Apply fully connected layers
        return x


# Load the model

def preprocess(title, text, subject, date):
    # Step 1 : Checking punctuation
    def check_punctuation(headline):
        issues = proselint.tools.lint(headline)
        print("checking")
        if(not len(issues)):
            return 1
        else:
            return 0
    
    # Step 2 : title length, text length, text sentence length
    stop_words = set(stopwords.words('english'))  # Get English stop words

    def title_length(title):
        # remove stop words and count the length of the title
        tit_len = len(title)
        title = title.split()
        title = [word for word in title if word.lower() not in stop_words]
        # count number of chars in title without stop words
        return [tit_len, len(title)]


    def text_length(text):
        # remove stop words and count the length of the text
        tex_len = len(text)
        text = text.split()
        text = [word for word in text if word.lower() not in stop_words]
        return [tex_len, len(text)]
    
    def text_sentence_count(text):
        # count number of sentences in the text
        # return text.fillna("").apply(lambda x: len(nltk.sent_tokenize(x)))
        return len(nltk.sent_tokenize(text))
    
    # Step 3 : Sentiment Analysis
    def sentiment_analyzer_scores(text):
        analyser = SentimentIntensityAnalyzer()
        score = analyser.polarity_scores(text)
        return [score['neg'], score['neu'], score['pos'], score['compound']]
    
    # Step 4 : Cateory ID
    def category_id(subject):
        categories = ['politicsNews', 'Government News', 'left-news', 'politics', 'worldnews', 'News', 'Middle-east', 'US_News']
        return categories.index(subject) + 1
    
    # Step 5 : Keyword Density
    def title_keyword_density(title):
        headline = title.strip().lower()
        tokens = word_tokenize(headline)
        tags = pos_tag(tokens)

        jj_count = sum(1 for word, tag in tags if tag == 'JJ')
        vbg_count = sum(1 for word, tag in tags if tag == 'VBG')
        rb_count = sum(1 for word, tag in tags if tag == 'RB')
        total_count = len(tokens)

        jj_density = (jj_count / total_count) * 100 if total_count > 0 else 0
        vbg_density = (vbg_count / total_count) * 100 if total_count > 0 else 0
        rb_density = (rb_count / total_count) * 100 if total_count > 0 else 0

        return [jj_density, vbg_density, rb_density]

    def text_keyword_density(text):
        headline = text.strip().lower()
        tokens = word_tokenize(headline)
        tags = pos_tag(tokens)

        jj_count = sum(1 for word, tag in tags if tag == 'JJ')
        vbg_count = sum(1 for word, tag in tags if tag == 'VBG')
        rb_count = sum(1 for word, tag in tags if tag == 'RB')
        total_count = len(tokens)

        jj_density = (jj_count / total_count) * 100 if total_count > 0 else 0
        vbg_density = (vbg_count / total_count) * 100 if total_count > 0 else 0
        rb_density = (rb_count / total_count) * 100 if total_count > 0 else 0

        return [jj_density, vbg_density, rb_density]
    
    return [category_id(subject), *title_length(title), *text_length(text), text_sentence_count(text), *sentiment_analyzer_scores(text), check_punctuation(title), *title_keyword_density(title), *text_keyword_density(text)]


def predict(input):
    input = torch.tensor(input).float().unsqueeze(0)
    model = torch.load('model_complete.pth', weights_only=False)
    model.eval()
    with torch.no_grad():
        outputs = model(input)
        probabilities = F.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
    return predictions.item()

# test_title = input("Enter the title of the article: ")
# test_text = input("Enter the text of the article: ")
# test_subject = input("Enter the subject of the article: ")
# test_date = input("Enter the date of the article: ")

# # Preprocess and convert input
# test_input = preprocess(test_title, test_text, test_subject, test_date)

# print("-------------------------------------------------------------------------------------------------------------------------------------------------------------")
# print("Tensor: ", test_input)
# print("-------------------------------------------------------------------------------------------------------------------------------------------------------------")
# print("\nThe length of the tensor: ", len(test_input))


# test_input = torch.tensor(test_input).float().unsqueeze(0)



# with torch.no_grad():  # Disable gradients for inference
#     # Model prediction
#     outputs = model(test_input)  # Logits (raw scores)
#     probabilities = F.softmax(outputs, dim=1)  # Convert logits to probabilities
#     predictions = torch.argmax(probabilities, dim=1)  # Get predicted class index

# # Print results
# print("\nLogits (Raw Output):", outputs.detach().numpy())
# print("Probabilities:", probabilities.detach().numpy())

# class_str = ""

# if(predictions.item() == 0):
#     class_str = "fake"
# else:
#     class_str = "real"

# print("\nPredicted Class:", predictions.item(), " ->  Class : ", class_str)  # Extract integer class