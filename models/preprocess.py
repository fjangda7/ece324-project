from utils import getTags
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import pandas as pd
from nltk.corpus import stopwords
import re

def plotToWords(raw_plot):
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
    data = {'figure': formattedReviews_trained, 'tags': tag}
    df = pd.DataFrame(data)

    return df