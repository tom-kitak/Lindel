import Lindel, os, sys
from Lindel.Predictor import *
import pickle as pkl
import utils.file_utils as file_utils
import pandas as pd
from tqdm import tqdm


def pearson_coefficient(dict1, dict2):
    # Get union of keys from both dictionaries
    keys = list(set(dict1.keys()) | set(dict2.keys()))

    # For each key, get the value from each dictionary or 0 if the key doesn't exist
    x = np.array([dict1.get(key, 0) for key in keys])
    y = np.array([dict2.get(key, 0) for key in keys])

    r = np.corrcoef(x, y)[0, 1]

    return r


if __name__ == "__main__":
    weights = pkl.load(open(os.path.join(Lindel.__path__[0], "Model_weights.pkl"), 'rb'))
    prerequesites = pkl.load(open(os.path.join(Lindel.__path__[0], 'model_prereq.pkl'), 'rb'))
    centered_ts_path = os.path.join(
        file_utils.get_root_directory(),
        "data",
        "local",
        "FORECasT",
        f"target_sequences_centered_60length.csv",
    )

    ts_df = pd.read_csv(centered_ts_path)
    coefficients = []

    freq_path = os.path.join(file_utils.get_root_directory(), "data", "local", "FORECasT", "freq")
    files = os.listdir(freq_path)
    for file in tqdm(files, total=len(files)):
        oligo_nr = file[:-4]
        label_df = pd.read_csv(os.path.join(freq_path, file))

        label = label_df.set_index('InsSeq')['frequency'].to_dict()
        seq = ts_df[ts_df['ID'] == 'Oligo'+oligo_nr]["TargetSequence"].iloc[0]

        y_hat, _ = gen_prediction(seq, weights, prerequesites)
        rev_index = prerequesites[1]
        pred_freq = {}
        total = 0.0
        for i in range(len(y_hat)):
            if any(nucleotide in rev_index[i] for nucleotide in ['A','T','C','G']) and (rev_index[i][0] == '1' or rev_index[i][0] == '2'):
                pred_freq[rev_index[i][2:]] = y_hat[i]
                total += y_hat[i]

        # Renormalize
        for k in pred_freq:
            pred_freq[k] = pred_freq[k] / total

        coefficients.append(pearson_coefficient(pred_freq, label))

    print(np.mean(np.array(coefficients)))
