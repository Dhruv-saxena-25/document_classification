from flask import Flask,request,render_template, redirect, url_for
from transformers import  AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
from werkzeug.utils import secure_filename
import re
import os
import nltk
import string
from nltk.corpus import stopwords
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")



def clean_text(file):
    ## Reading the document.
    with open(file, encoding='utf8') as f:
        for text in f:
             text =text.strip()
             print(text)
    ## Applying text-processing on text.
    text = str(text).lower()
    text = re.sub("\[.*?\]", '', text)
    text = re.sub("https?://\S+|www\.\S+", '', text)
    text = re.sub("<.*?>+", '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub("\n", '', text)
    text = re.sub("\w*\d\w*", '', text)
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
    return render_template('index.html')


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
            os.makedirs("text", exist_ok= True)
            # Extract the text input from the form
            file = request.files['file']
            if file:
                file_path = os.path.join("text/" + TIMESTAMP + secure_filename(file.filename))
                file.save(file_path)
            test = clean_text(file_path)
            token_fine = AutoTokenizer.from_pretrained("./notebook\data")
            model_fine = AutoModelForSequenceClassification.from_pretrained("./notebook\data")
            pipe = pipeline("text-classification", model= model_fine, tokenizer= token_fine)
            label_mapping = {'LABEL_0': "Sports", 'LABEL_1': "Business", 'LABEL_2': "Politics",
                 'LABEL_3': "Technology", 'LABEL_4': "Entertainment"}
            pred = f"{label_mapping[pipe(test)[0]['label']]}"
            print(label_mapping[pipe(test)[0]['label']])
            return render_template('index.html',results=pred)

if __name__=="__main__":
     app.run(host="0.0.0.0", port=8080, debug= True)


