import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from filter_and_center_target_sequences import filter_and_center_target_sequences
from generate_labels import generate_labels


def data_preprocessing(settings):
    print("Current settings:")
    print(settings)

    # Filter and center target sequences
    filter_and_center_target_sequences(settings)

    generate_labels(settings)

    print("Finished data preprocessing")


if __name__ == "__main__":
    settings = {
        "window_length": 60,
        "prediction_type": "insertions",  # options: 'deletions', 'insertions' or 'indels'
    }
    data_preprocessing(settings)
