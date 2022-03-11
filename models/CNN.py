import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from keras import *
from keras.layers import *
from keras.utils import to_categorical

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import random

df=pd.read_csv("wiki_movie_plots_deduped.csv")

genres=['romance','comedy']
df=df[df['Genre'].isin(genres)]
df=df.reset_index()
df['GenreID']=df['Genre'].apply(lambda x: genres.index(x))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(df['Plot']))
sequences = tokenizer.texts_to_sequences(list(df['Plot']))
maxSequenceLength=np.max([len(sequence) for sequence in sequences])
indexOfWord = tokenizer.indexOfWord
data = pad_sequences(sequences, maxlen=maxSequenceLength)

embedding={}
index=0
with open('glove.6B.100d.txt') as file:
    for i in file:
        line=i.split()
        coefs = np.asarray(line[1:], dtype='float32')
        embedding[line[0]]=coefs
        index+=1

matrix=np.zeros((len(indexOfWord)+1,len(embedding['a'])))
for word,i in indexOfWord.items():
    if word in embedding:
        matrix[i]=embedding[word]


embeddingLayer = Embedding(len(indexOfWord) + 1,
                            len(embedding['a']),
                            weights=[matrix],
                            input_length=maxSequenceLength,
                            trainable=False)

inputOfSequence = Input(shape=(maxSequenceLength,), dtype='int32')
emSequence = embeddingLayer(inputOfSequence)
x=Conv1D(128, 9, activation='relu')(emSequence)
x=MaxPooling1D(9)(x)
x=Conv1D(128, 9, activation='relu')(x)
x = Dropout(.4)(x)
x=MaxPooling1D(9)(x)
x=Conv1D(128, 9, activation='relu')(x)
x = Dropout(.4)(x)
x=MaxPooling1D(9)(x)

x = Flatten()(x)
x=Dense(128, activation='relu')(x)
x=Dense(len(genres),activation='softmax')(x)

cnn = Model(inputOfSequence, x)
cnn.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

randGen=random.randint(1,1000)
X_train, X_test, y_train, y_test = train_test_split(data, df['GenreID'], test_size=0.2, random_state=randGen)
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

X_train.shape
cnn.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=2, batch_size=128)