from flask import Flask,request,render_template, redirect, url_for
import numpy as np
import pandas as pd
import re
import os
import string
import nltk
from keras.utils import pad_sequences
from werkzeug.utils import secure_filename
from nltk.corpus import stopwords
nltk.download('stopwords')
import keras
import pickle
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


load_model=keras.models.load_model("notebook\\model.pkl")
with open('notebook\\tokenizer.pickle', 'rb') as handle:
    load_tokenizer = pickle.load(handle)


# Let's apply stemming and stopwords on the data
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))


def clean_text(file):
    with open(file, 'r', encoding='utf8') as f:
        for text in f:
             text = text.strip()
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    print(text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text


application=Flask(__name__)
app=application


## Route for a home page

@app.route('/')
def index():
    return render_template('home.html')


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
            # Extract the text input from the form
            file = request.files['file']
            if file:
                file_path = os.path.join("data/" + TIMESTAMP + secure_filename(file.filename))
                file.save(file_path)
            test = clean_text(file_path)
            seq = load_tokenizer.texts_to_sequences(test)
            padded = pad_sequences(seq, maxlen=3000)
            pred = load_model.predict(padded)
            pred = np.argmax(pred[0])
            label_mapping = {0: "Sport", 1: "Business", 2: "Politics", 3: "Tech", 4: "Entertainment"}
            return render_template('home.html',results=label_mapping[pred])



if __name__=="__main__":
     app.run(host="0.0.0.0", port=8080, debug= True)

     