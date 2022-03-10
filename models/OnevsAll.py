#need to edit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib
X_train = np.array(x_train['preprocessed_data'])
Y_train = np.array(x_train['tags'])
X_test = np.array(x_test['preprocessed_data'])
Y_test = np.array(x_test['tags'])
model = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', OneVsRestClassifier(SGDClassifier(loss='log', penalty='l1', class_weight="balanced"), n_jobs=-1))])
model.fit(X_train, Y_train)
joblib.dump(model, "pipeline1.pkl", compress=9)
prediction = model.predict(X_test)

precision = precision_score(y_test, predictions, average='micro')
recall = recall_score(y_test, predictions, average='micro')
f1 = f1_score(y_test, predictions, average='micro')
 
print("Micro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

precision = precision_score(y_test, predictions, average='macro')
recall = recall_score(y_test, predictions, average='macro')
f1 = f1_score(y_test, predictions, average='macro')
 
print("Macro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
print (metrics.classification_report(y_test, predictions))