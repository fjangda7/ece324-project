#KNN

import nltk
nltk.download('stopwords')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
import pandas as pd
from bs4 import BeautifulSoup
import re

from utils import predict
from utils import getTags


def plotToWords(raw_plot):
    global stopwords
    figure = BeautifulSoup(raw_plot, "lxml")
    justLetters = re.sub("[^a-zA-Z]", " ", figure.get_text())
    lower = justLetters.lower()
    words = lower.split()
    uselessWords = set(stopwords.words("english"))
    keys = [w for w in words if not w in uselessWords]
    return (" ".join(keys))

def preprocess(filename):
    train = pd.read_csv(filename)

    numberOfReviews = train["Plot"].size
    formattedReviews_trained = []

    for i in range(numberOfReviews):
        formattedReviews_trained.append(plotToWords(train["Plot"][i]))

    tag = getTags('Comedy', train)
    data = {'plot': formattedReviews_trained, 'tags': tag}
    df = pd.DataFrame(data)

    return df

csvFileName = 'trainingSet.csv'
features = preprocess('trainingSet.csv')

train, test = train_test_split(features, test_size=0.1, random_state=42)

tfidf = TfidfVectorizer(min_df=2, tokenizer=None, preprocessor=None, stop_words=None)
featuresTrained = tfidf.fit_transform(train['plot'])
featuresTrained = featuresTrained.toarray()

knn = KNeighborsClassifier(n_neighbors=3, n_jobs=1, algorithm='brute', metric='cosine')
knn = knn.fit(featuresTrained, train['tags'])

predict(tfidf, knn, test)