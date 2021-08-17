import pandas as pd
from pandas.core import base
import numpy as np
import nltk
import re
import pickle
import string
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

base_path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep

def train():
    passive_aggressive = PassiveAggressiveClassifier(max_iter=130)

    train_dataframe = pd.read_csv(base_path + "fake_news_training_dataset.csv").rename(columns = {'title': 'Headline', 'author': 'Writer','text':'news_info','label':'Real'}, inplace = False).dropna()
    output_train=train_dataframe['Real']
    input_train=train_dataframe.drop('Real',axis=1)
    input_train.reset_index(inplace=True)

    lemma = WordNetLemmatizer()
    corpus = []

    for i in range(len(input_train)):
        print(i)
        data = re.sub("<.*?>+", "", input_train["news_info"][i])
        data = re.sub(
            "[%s]" % re.escape(string.punctuation), "", input_train["news_info"][i]
        )
        data = re.sub("\[.*?\]", "", input_train["news_info"][i])
        data = re.sub("\w*\d\w*", "", input_train["news_info"][i])
        data = data.lower()
        data = data.split()
        data = [
            lemma.lemmatize(word)
            for word in data
            if not word in set(stopwords.words("english"))
        ]
        data = " ".join(data)
        corpus.append(data)

    count_vec = CountVectorizer(max_features=600, ngram_range=(1, 5))
    input_transformed = count_vec.fit_transform(corpus).toarray()
    pickle.dump(count_vec, open(base_path + "count_vec.pkl", "wb"))

    X_TRAIN2, X_TEST2, Y_TRAIN2, Y_TEST2 = train_test_split(
        input_transformed, output_train, test_size=0.30, random_state=250
    )

    passive_aggressive.fit(X_TRAIN2, Y_TRAIN2)
    pickle.dump(passive_aggressive, open(base_path + "passive_aggressive.pkl", "wb"))

    y_pred = passive_aggressive.predict(X_TEST2)
    score = accuracy_score(Y_TEST2, y_pred)

    print(f"Accuracy: {round(score*100,2)}%")


def predict(news):
    input_data = [news]

    count_vec = pickle.load(open(base_path + "count_vec.pkl", "rb"))
    passive_aggressive = pickle.load(open(base_path + "passive_aggressive.pkl", "rb"))

    vectorized_input_data = count_vec.fit_transform(input_data)
    prediction = passive_aggressive.predict(vectorized_input_data)[0]

    return prediction