from keras.models import load_model
import numpy as np


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

# Load your test data
# Replace 'path_to_your_test_data' with the actual path to your test data file
test_data_path = 'C:\\Users\\tomaz\\OneDrive\\Namizje\\AI4CRISPR\\Lindel\\Lindel_data\\Lindel_test.txt'
test_data = np.loadtxt(test_data_path, delimiter="\t", dtype=str)

# Assuming the first column is sequence data and the rest are numerical features
Seqs_test = test_data[:, 0]
X_test_numerical = test_data[:, 1:].astype('float32')

# One-hot encode your sequence data
x_test_encoded = [onehotencoder(seq) for seq in Seqs_test]

# Assuming your model expects only the one-hot encoded sequence data
x_test_encoded = np.array(x_test_encoded)

# Load the model
# Replace this with the actual path to your model
model_path = "C:\\Users\\tomaz\\OneDrive\\Namizje\\AI4CRISPR\\Lindel\\Lindel_data\\saved\\L1_indel.h5"
model = load_model(model_path)

# Make predictions
predictions = model.predict(x_test_encoded)

# Output the predictions
print(predictions)
