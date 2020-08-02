#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import pandas as pd
from bs4 import BeautifulSoup
import re
import os
import pickle
from tensorflow import keras
import tensorflow_datasets as tfds


def load_model(path):
    return tf.keras.models.load_model(path)


def clean_tweet(tweet):
    tweet = BeautifulSoup(tweet, "lxml").get_text()
    # Removing the @
    tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet)
    # Removing the URL links
    tweet = re.sub(r"https?://[A-Za-z0-9./]+", ' ', tweet)
    # Keeping only letters
    tweet = re.sub(r"[^a-zA-Z.!?']", ' ', tweet)
    # Removing additional whitespaces
    tweet = re.sub(r" +", ' ', tweet)
    return tweet


# ### Loading tokenizer variable if exists
def tokenizer_loadCreate(tokenizer_path):
    tokenizer_save = os.path.join(tokenizer_path, 'tokenizer.pickle')
    if os.path.exists(tokenizer_save):
        print('[!!] Loading tokenizer data')
        with open(tokenizer_save, 'rb') as load_file:
            tokenizer = pickle.load(load_file)
            print('[!!] Tokenizer_data loaded')
            return tokenizer


def build_new_model():
    # ## Load data tweets data set

    train_path = 'training_data/train.csv'
    test_path = 'training_data/test.csv'
    save_model_path = 'model_save'
    tokenizer_path = 'output'

    print('[!!] Loading data ...')
    cols = ["sentiment", "id", "date", "query", "user", "text"]
    train_data = pd.read_csv(
        train_path,
        header=None,
        names=cols,
        engine="python",
        encoding="latin1"
    )
    test_data = pd.read_csv(
        test_path,
        header=None,
        names=cols,
        engine="python",
        encoding="latin1"
    )

    print('Train data head')
    print(train_data.head())

    print('Test data head')
    print(test_data.head())

    data = train_data  # using only the train data set

    # Preprocessing
    # Cleaning
    data.drop(["id", "date", "query", "user"],
              axis=1,
              inplace=True)

    # Clean tweet function here

    # ### Loading data_clean variable if exists else create one

    data_clean_save = os.path.join(tokenizer_path, 'data_clean.pickle')
    if os.path.exists(data_clean_save):
        print('[!!] Loading data clean')
        with open(data_clean_save, 'rb') as load_file:
            data_clean = pickle.load(load_file)
        print('[!!] Data_clean loaded [!!]')
    else:
        print('[!!] Cleaning data using clean_tweet function ...')
        data_clean = [clean_tweet(tweet) for tweet in data.text]
        with open(data_clean_save, 'wb') as save_file:
            pickle.dump(data_clean, save_file)

    print(data_clean[:5])
    data_labels = data.sentiment.values
    data_labels[data_labels == 4] = 1
    print('Data labels ..')
    print(data_labels)

    # ### Loading tokenizer variable if exists
    def tokenizer_loadCreate():
        tokenizer_save = os.path.join(tokenizer_path, 'tokenizer.pickle')
        if os.path.exists(tokenizer_save):
            print('[!!] Loading tokenizer data')
            with open(tokenizer_save, 'rb') as load_file:
                tokenizer = pickle.load(load_file)
                print('[!!] Tokenizer_data loaded')
                return tokenizer
        else:
            print('[!!] Tokenizer not found Creating one and saving to pickle ...')
            tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                data_clean, target_vocab_size=2 ** 16
            )
            with open(tokenizer_save, 'wb') as save_file:
                print(f'[!!] Saving tokenizer to : {tokenizer_save}')
                pickle.dump(tokenizer, save_file)
            return tokenizer

    tokenizer = tokenizer_loadCreate()

    # ### Loading data_inputs variable
    def data_inputs_loadCreate():
        data_inputs_save = os.path.join(tokenizer_path, 'data_inputs.pickle')
        if os.path.exists(data_inputs_save):
            print('[!!] Loading data input')
            with open(data_inputs_save, 'rb') as load_file:
                data_inputs = pickle.load(load_file)
            print('[!!] data_inputs loaded')
            return data_inputs
        else:
            print('[!!] Encoding data')
            data_inputs = [tokenizer.encode(sentence) for sentence in data_clean]
            with open(data_inputs_save, 'wb') as save_file:
                pickle.dump(data_inputs, save_file)
            return data_inputs

    data_inputs = data_inputs_loadCreate()

    # ### Padding
    print('[!!] Padding ..')
    MAX_LEN = max([len(sentence) for sentence in data_inputs])
    data_inputs = tf.keras.preprocessing.sequence.pad_sequences(data_inputs,
                                                                value=0,
                                                                padding="post",
                                                                maxlen=MAX_LEN)

    # ### Splitting the training/testing set
    print('[!!] Splitting the traning/testing set')
    test_idx = np.random.randint(0, 800000, 8000)
    test_idx = np.concatenate((test_idx, test_idx + 800000))
    test_inputs = data_inputs[test_idx]
    test_labels = data_labels[test_idx]
    train_inputs = np.delete(data_inputs, test_idx, axis=0)
    train_labels = np.delete(data_labels, test_idx)

    VOCAB_SIZE = tokenizer.vocab_size
    EMB_DIM = 200
    NB_FILTERS = 100
    FFN_UNITS = 256
    NB_CLASSES = len(set(train_labels))

    DROPOUT_RATE = 0.2

    BATCH_SIZE = 32
    NB_EPOCHS = 1  # 5

    # ## Define model
    def define_model():
        model = keras.Sequential()
        model.add(keras.layers.Embedding(VOCAB_SIZE, EMB_DIM))
        model.add(keras.layers.Conv1D(filters=NB_FILTERS, kernel_size=2, padding='valid', activation='relu'))
        model.add(keras.layers.GlobalMaxPool1D())
        model.add(keras.layers.Dense(units=FFN_UNITS, activation='relu'))
        model.add(keras.layers.Dropout(rate=DROPOUT_RATE))
        model.add(keras.layers.Dense(units=1, activation='sigmoid'))  # Last layer
        model.summary()
        # ### Compile
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    model = define_model()
    # ### Fit
    model.fit(train_inputs, train_labels, batch_size=BATCH_SIZE, epochs=NB_EPOCHS)

    model.predict(np.array([tokenizer.encode("I have excellent apartment")]))

    test = np.array([tokenizer.encode("You are so funny")])

    model.predict(test)

    model.save('cnn_model.h5')

    loaded_model = tf.keras.models.load_model('cnn_model.h5')

    loaded_model.summary()

    loaded_model.predict(
        np.array([tokenizer.encode("we're close to the peak of coronavirus right? this shit sucks LOL")]))

m = load_model('/Users/tzvip/PycharmProjects/COVID19-SIR/sentiment/cnn_model.h5')
m.summary()