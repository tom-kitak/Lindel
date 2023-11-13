import numpy as np
from keras.models import load_model
import pickle as pkl
import pandas as pd
from tqdm import tqdm


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


if __name__ == "__main__":

    settings = {
        "window_length": 8,
        "save_model_folder": "insertions_alibi_data\\",
        "data_dir": "C:\\Users\\tomaz\\OneDrive\\Namizje\\AI4CRISPR\\Lindel\\FORECasT_data\\",
        "labels_dir":
            "C:\\Users\\tomaz\\OneDrive\\Namizje\\AI4CRISPR\\Lindel\\FORECasT_data\\train_insertions_8length\\",
    }

    # Set up
    sequences_df = pd.read_csv(settings["data_dir"]
                               + f"target_sequences_explorative_train_centered_{settings['window_length']}length.csv")

    # Preprocess data
    X, y = [], []
    for index, row in tqdm(sequences_df.iterrows(), total=len(sequences_df)):
        oligo_id, sequence, _, _, _, _ = row
        try:
            labels = np.loadtxt(settings["labels_dir"] + f"label_{oligo_id[5:]}.csv")
        except FileNotFoundError:
            # File is missing, skip this target sequence
            continue
        X.append(onehotencoder(sequence))
        y.append(labels)

    X = np.array(X)
    y = np.array(y)

    # Load the model
    model = load_model(settings["data_dir"] + settings["save_model_folder"] + "L1_ins.h5")

    predictions = model.predict(X)
    pear_coeffs = []

    for p in range(len(predictions)):
        pear_coeffs.append(np.corrcoef(predictions[p], y[p])[0][1])

    pear_coeff = np.array(pear_coeffs).mean()
    print("Pearson Correlation Coefficient:", pear_coeff)
