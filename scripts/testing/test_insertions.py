import numpy as np
from keras.models import load_model
import pickle as pkl


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
    # Load the saved model
    workdir = "C:\\Users\\tomaz\\OneDrive\\Namizje\\AI4CRISPR\\Lindel\\Lindel_data\\"
    fname = "Lindel_test.txt"

    label, rev_index, features = pkl.load(open(workdir + 'feature_index_all.pkl', 'rb'))
    feature_size = len(features) + 384
    data = np.loadtxt(workdir + fname, delimiter="\t", dtype=str)
    Seqs = data[:, 0]
    data = data[:, 1:].astype('float32')

    # Sum up deletions and insertions to
    X = data[:, :feature_size]
    y = data[:, feature_size:]

    Seq_train = Seqs
    x_test, y_test = [], []
    for i in range(len(y)):
        if 1 > sum(y[i, -21:]) > 0:  # 5 is a random number i picked if i use pred_size here it will be -21:0 it will just generate empty array
            y_test.append(y[i, -21:] / sum(y[i, -21:]))
            x_test.append(onehotencoder(Seq_train[i][-6:]))

    x_test, y_test = np.array(x_test), np.array(y_test)

    # Load the model
    model = load_model(workdir + 'save_insertion\\L1_ins.h5')

    predictions = model.predict(x_test)
    pear_coeffs = []

    for p in range(len(predictions)):
        pear_coeffs.append(np.corrcoef(predictions[p], y_test[p])[0][1])

    pear_coeff = np.array(pear_coeffs).mean()
    print("Pearson Correlation Coefficient:", pear_coeff)
