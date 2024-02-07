import keras
import pandas as pd
from nltk.stem.isri import ISRIStemmer
from keras.datasets import imdb
import string
import re
import requests
from transformers import pipeline
from rank_bm25 import BM25Okapi
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from pyarabic.araby import tokenize, is_arabicrange,strip_diacritics
import nltk
import os




API_URL = "https://api-inference.huggingface.co/models/hemagamal/mdeberta_Quran_qa"
headers = {"Authorization": "Bearer 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"}



dir_path = '/home/ibrahim/python_code/NLP/Open_domain_QA/data'
# Get a list of the filenames in the directory
file_names = os.listdir(dir_path)
# Loop over the filenames and read each file
corpus=[]
for file_name in file_names:
    file_path = os.path.join(dir_path, file_name)
    with open(file_path, 'r') as f:
        # Read the entire contents of the file into a string variable
        lines=f.readlines()
        for i in range(0,len(lines)):
            corpus.append(lines[i])


arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ-\n٪'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations
def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("گ", "ك", text)
    return text
def preprocess(text):
    st=ISRIStemmer()
    regex = re.compile(r"(http|https|ftp)://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    review = re.sub(regex, '', text)
    translator = str.maketrans('', '', punctuations_list)
    review = review.translate(translator) # remove ÷×؛<>_()*&^%][ـ،/:"؟
    review = tokenize(review, is_arabicrange,morphs=strip_diacritics)
    review = [st.stem(word) for word in review if not word in set(stopwords.words('arabic'))]
    text = ' '.join(review)
    review=normalize_arabic(text)
    return review


cleand_corpus = squared_list = list(map(preprocess, corpus))


def bm25(question,corpus,cleand_corpus,top_answer):
  tokenized_corpus = [tokenize(doc) for doc in cleand_corpus]
  bm25 = BM25Okapi(tokenized_corpus)
  query=preprocess(question)
  tokenized_query = tokenize(query)
  top_corpus=bm25.get_top_n(tokenized_query, corpus, n=top_answer)
  return top_corpus


from flask import Flask, request, jsonify,render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_answer():
    data = request.json
    question = data["question"]
    response = requests.post(API_URL, headers=headers, json={"context": bm25(question,corpus,cleand_corpus,3)[0]
                                                             , "question": question})
    if(len(response.json()['answer'])!=0):
        print(bm25(question,corpus,cleand_corpus,3)[0:5])
        return jsonify({"answer": response.json()['answer'], "context": bm25(question,corpus,cleand_corpus,3)[0]})
    else:
        print("waittttt")


app.run()






