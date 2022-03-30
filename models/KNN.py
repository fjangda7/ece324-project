#KNN

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import nltk
nltk.download()
from utils import predict
from preprocess import preprocess


csvFileName = 'trainingSet.csv'
features = preprocess('trainingSet.csv')

train, test = train_test_split(features, test_size=0.1, random_state=42)

tfidf = TfidfVectorizer(min_df=2, tokenizer=None, preprocessor=None, stop_words=None)
featuresTrained = tfidf.fit_transform(train['plot'])
featuresTrained = featuresTrained.toarray()

knn = KNeighborsClassifier(n_neighbors=3, n_jobs=1, algorithm='brute', metric='cosine')
knn = knn.fit(featuresTrained, train['tags'])

predict(tfidf, knn, test)