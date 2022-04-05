from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

tags = ['comedy', 'others']

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(tags))
    targets = tags
    plt.xticks(ticks, targets, rotation=45)
    plt.yticks(ticks, targets)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def evaluate_prediction(predictions, target, title="Confusion Matrix"):
    print('accuracy %s' % accuracy_score(target, predictions))
    print('precision %s' % precision_score(target, predictions,pos_label='Comedy'))
    print('recall %s' % recall_score(target, predictions,pos_label='Comedy'))
    print('f-measure %s' % f1_score(target, predictions,pos_label='Comedy'))

    cm = confusion_matrix(target, predictions)
    print('confusion matrix\n %s' % cm)
    print('(row=expected, col=predicted)')

    normalizedMatrix = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plot_confusion_matrix(normalizedMatrix, title + ' Normalized')
    plt.show()


def predict(vectorizer, classifier, data):
    features = vectorizer.transform(data['plot'])
    predictions = classifier.predict(features)
    target = data['tags']
    evaluate_prediction(predictions, target)


def getTags(genre, train):
    tags = []
    for tag in train['Genre1']:
        if tag == genre:
            tags.append(genre)
        else:
            tags.append('other')

    return tags