from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import re
import string
import nltk
from keras.utils import pad_sequences
from nltk.corpus import stopwords
nltk.download('stopwords')
import keras
import pickle

load_model=keras.models.load_model("notebook\\model.pkl")
with open('notebook\\tokenizer.pickle', 'rb') as handle:
    load_tokenizer = pickle.load(handle)


# Let's apply stemming and stopwords on the data
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))


def clean_text(text):
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
            text=request.form.get('text')
            test = clean_text(text)
            seq = load_tokenizer.texts_to_sequences(test)
            padded = pad_sequences(seq, maxlen=3000)
            pred = load_model.predict(padded)
            pred = np.argmax(pred[0]) 
            label_mapping = {0: "Sport", 1: "Business", 2: "politics", 3: "tech", 4: "entertainment"}
            return render_template('home.html',results= label_mapping[pred])



if __name__=="__main__":
     app.run(host="0.0.0.0", port=8080, debug= True)