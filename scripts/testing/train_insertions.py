#!/usr/bin/env python

#System tools
import pickle as pkl
import os,sys,csv,re

from tqdm import tqdm_notebook as tqdm
import pylab as pl
import numpy as np

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, Flatten
from keras.models import Sequential, load_model
from keras.regularizers import l2, l1
import pandas as pd
from tqdm import tqdm
import numpy as np


# Define useful functions
def mse(x, y):
    return ((x-y)**2).mean()

def corr(x, y):
    return np.corrcoef(x, y)[0, 1] ** 2


def onehotencoder(seq):
    nt= ['A','T','C','G']
    head = []
    l = len(seq)
    for k in range(l):
        for i in range(4):
            head.append(nt[i]+str(k))

    for k in range(l-1):
        for i in range(4):
            for j in range(4):
                head.append(nt[i]+nt[j]+str(k))
    head_idx = {}
    for idx,key in enumerate(head):
        head_idx[key] = idx
    encode = np.zeros(len(head_idx))
    for j in range(l):
        encode[head_idx[seq[j]+str(j)]] =1.
    for k in range(l-1):
        encode[head_idx[seq[k:k+2]+str(k)]] =1.
    return encode


def ensure_directory_exists(directory_path):
    """Ensure that a directory exists. If it doesn't, create it."""
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Directory {directory_path} created.")
        except OSError as e:
            print(f"Error creating directory {directory_path}: {e}")


if __name__ == "__main__":
    settings = {
        "window_length": 52,
        "save_model_folder": "insertions_52wl_alibi_data\\",
        "validation_set": 0.1,  # Validation set percentage
        "num_outcomes": 20,
    }
    DATA_DIR = "C:\\Users\\tomaz\\OneDrive\\Namizje\\AI4CRISPR\\Lindel\\FORECasT_data\\"
    LABELS_DIR = DATA_DIR + f"train_insertions_{settings['window_length']}length\\"

    # Train code
    ensure_directory_exists(DATA_DIR + settings["save_model_folder"])

    # Set up
    sequences_df = pd.read_csv(DATA_DIR
                               + f"target_sequences_explorative_train_centered_{settings['window_length']}length.csv")

    # Preprocess data
    x_train, y_train = [], []
    for index, row in tqdm(sequences_df.iterrows(), total=len(sequences_df)):
        oligo_id, sequence, _, _, _, _ = row
        try:
            labels = np.loadtxt(LABELS_DIR + f"label_{oligo_id[5:]}.csv")
        except FileNotFoundError:
            # File is missing, skip this target sequence
            continue
        x_train.append(onehotencoder(sequence))
        y_train.append(labels)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    np.random.seed(121)
    idx = np.arange(len(y_train))
    np.random.shuffle(idx)
    x_test, y_test = x_train[idx], y_train[idx]
    valid_size = round(len(x_test) * settings["validation_set"])

    x_valid = x_train[:valid_size]
    x_train = x_train[valid_size:]

    y_valid = y_train[:valid_size]
    y_train = y_train[valid_size:]

    size_input = x_train.shape[1]
    num_outcomes = y_train.shape[1]  # Sus

    # Train model
    lambdas = 10 ** np.arange(-10, -1, 0.1)  # for activation function test
    errors_l1, errors_l2 = [], []
    for l in tqdm(lambdas):
        np.random.seed(0)
        model = Sequential()
        model.add(Dense(num_outcomes, activation='softmax', input_shape=(size_input,), kernel_regularizer=l2(l)))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
        model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
                  callbacks=[EarlyStopping(patience=1)], verbose=0)
        y_hat = model.predict(x_valid)
        errors_l2.append(mse(y_hat, y_valid))

    for l in tqdm(lambdas):
        np.random.seed(0)
        model = Sequential()
        model.add(Dense(num_outcomes, activation='softmax', input_shape=(size_input,), kernel_regularizer=l1(l)))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
        model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
                  callbacks=[EarlyStopping(patience=1)], verbose=0)
        y_hat = model.predict(x_valid)
        errors_l1.append(mse(y_hat, y_valid))

    np.save(DATA_DIR + settings["save_model_folder"] + "mse_l1_ins.npy", errors_l1)
    np.save(DATA_DIR + settings["save_model_folder"] + "mse_l2_ins.npy", errors_l2)

    # final model
    l = lambdas[np.argmin(errors_l1)]
    np.random.seed(0)
    model = Sequential()
    model.add(Dense(num_outcomes, activation='softmax', input_shape=(size_input,), kernel_regularizer=l1(l)))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
                        callbacks=[EarlyStopping(patience=1)], verbose=0)

    model.save(DATA_DIR + settings["save_model_folder"] + "L1_ins.h5")

    l = lambdas[np.argmin(errors_l2)]
    np.random.seed(0)
    model = Sequential()
    model.add(Dense(num_outcomes, activation='softmax', input_shape=(size_input,), kernel_regularizer=l2(l)))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
                        callbacks=[EarlyStopping(patience=1)], verbose=0)

    model.save(DATA_DIR + settings["save_model_folder"] + "L2_ins.h5")


