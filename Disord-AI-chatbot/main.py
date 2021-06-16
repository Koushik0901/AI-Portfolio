# import all the libraries
import json 
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

#open the intents file
with open('intents.json') as file:
    data = json.load(file)
    
training_sentences = []
training_labels = []
labels = []
responses = []

#if chat_model.h5 already exist don't retrain
try:
    model = keras.models.load_model('./chat_model.h5')
    print("Already Trained!")
except:
    for intent in data['intents']:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)#appending the patterns
            training_labels.append(intent['tag'])#appending the tag
        responses.append(intent['responses'])#appending the responses
        
        if intent['tag'] not in labels:
            labels.append(intent['tag'])#creating labels
        
    num_classes = len(labels)#number of classes

    lbl_encoder = LabelEncoder()
    lbl_encoder.fit(training_labels)
    training_labels = lbl_encoder.transform(training_labels)

    vocab_size = 1000
    embedding_dim = 128
    max_len = 20
    oov_token = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(training_sentences)#tokenizing sentences
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)#padding the sentences

    #simple callback class that stops the training if the accuracy reaches 97.5%
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy')>0.975):
                print("\n\nReached 97.5% accuracy so cancelling training!\n")
                self.model.stop_training = True

    callbacks = myCallback()

    #creating a bidirectional lstm model
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', 
                optimizer='adam', metrics=['accuracy'])

    model.summary()

    epochs = 100
    history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs, callbacks=[callbacks])

    # to save the trained model
    model.save("chat_model.h5")

    import pickle

    # to save the fitted tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # to save the fitted label encoder
    with open('label_encoder.pickle', 'wb') as ecn_file:
        pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
