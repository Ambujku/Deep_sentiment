import pandas as pd
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, Flatten

from keras.datasets import imdb 
top_words = 5000 
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

word_to_id = keras.datasets.imdb.get_word_index()

word_to_id["<PAD>"] = 0

word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

id_to_word = {value:key for key,value in word_to_id.items()}

from numpy import array

from keras.preprocessing.text import one_hot 

docs = ['Gut gemacht',
        'Gute arbeit',
        'Super idee',
        'Perfekt erledigt',
        'exzellent',
        'naja',
        'Schwache arbeit.',
        'Nicht gut',
        'Miese arbeit.',
        'Hätte es besser machen können.']

vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]


max_review_length = 500 
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

embedding_vector_length = 32
model = Sequential() 
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(Flatten())

#model.add(LSTM(100)) 
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=6, batch_size=64)
scores = model.evaluate(X_test, y_test, verbose=0) 
print("Accuracy: %.2f%%" % (scores[1]*100))


test_1 = "this movie was terrible and bad"
test_2 = "i really liked the movie and had fun"
for review in [test_2,test_1]:
    tmp = []
    for word in review.split(" "):
        tmp.append(word_to_id[word])
    tmp_padded = sequence.pad_sequences([tmp], maxlen=max_review_length) 
    print("%s. Sentiment: %s" % (review,model.predict(array([tmp_padded][0]))[0][0]))

