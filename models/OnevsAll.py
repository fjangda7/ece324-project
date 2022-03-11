from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import predict
from preprocess import preprocess

csvFileName = 'data/trainingSet.csv'
features = preprocess(csvFileName)
train, test = train_test_split(features, test_size=0.1, random_state=42)

tfidf = TfidfVectorizer(min_df=2, tokenizer=None, preprocessor=None, stop_words=None)

featuresTrained = tfidf.fit_transform(train['plot'])
featuresTrained = featuresTrained.toarray()

lin_clf = svm.LinearSVC()
lin_clf.fit(featuresTrained, train['tags'])

predict(tfidf, lin_clf, test)