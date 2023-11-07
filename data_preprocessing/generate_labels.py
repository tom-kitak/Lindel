import os
import pandas as pd
import numpy as np
import utils.file_utils as file_utils
from tqdm import tqdm


def generate_labels(settings):
    # Set up
    centered_ts_path = os.path.join(
        file_utils.get_root_directory(),
        "data",
        "local",
        "FORECasT",
        f"target_sequences_centered_{settings['window_length']}length.csv",
    )

    centered_ts_df = pd.read_csv(centered_ts_path)

    window_length = settings["window_length"]
    missing_files = []
    oow_counts = {  # For counting the number of outcomes that are out-of-window excluded
        "n_samples": 0,  # Total number of outcomes
        "n_del_start_oow": 0,  # Number of cases where the left edge of the deletion is out-of-window
        "n_del_end_oow": 0,  # Number of cases where the right edge of the deletion is out-of-window
    }
    ts_ids = []

    # Start processing
    for index, row in tqdm(centered_ts_df.iterrows(), total=len(centered_ts_df)):
        id, target_sequence, _, _, _ = row

        # Get files
        oligo_nr = int(id[5:])
        sorted_filename = id[:5] + "_" + str(oligo_nr) + ".tij.sorted.tsv"
        try:
            observed_outcomes_df = pd.read_table(
                os.path.join(
                    file_utils.get_sorted_outcomes_folder_path(), sorted_filename
                )
            )

            # Select outcomes you want to study
            if settings["prediction_type"] == "deletions":
                raise Exception("Not implemented")
            elif settings["prediction_type"] == "insertions":
                observed_outcomes_df = observed_outcomes_df[
                    observed_outcomes_df["Type"] == "INSERTION"
                ]

                # TODO: Remove when extending the program with more than 2 insertions
                size_2_condition_observed = observed_outcomes_df["Size"] == 2
                size_1_condition_observed = observed_outcomes_df["Size"] == 1
                observed_outcomes_df = observed_outcomes_df[
                    size_2_condition_observed | size_1_condition_observed
                ]
                # Start must be 0
                start_condition_observed = observed_outcomes_df["Start"] == 0
                observed_outcomes_df = observed_outcomes_df[start_condition_observed]

        except FileNotFoundError:
            # File is missing, skip this target sequence
            missing_files.append(oligo_nr)
            continue

        ts_middle = len(target_sequence) // 2  # Middle of the target sequence
        ts_window_extent = (
            window_length // 2
        )  # One-sided Length of the considered window
        window_edge_l = ts_middle - ts_window_extent  # Index of the start of the window
        window_edge_r = (
            ts_middle + ts_window_extent
        )  # Index of the first nt after the end of the window

        # Generate labels:
        # Frequencies for all possible outcomes per sequence

        total_observations = observed_outcomes_df["countEvents"].sum()
        if total_observations < 100:
            # Skip target sequences with < 100 mutagenic reads
            continue
        observed_outcomes_df["frequency"] = (
            observed_outcomes_df["countEvents"] / total_observations
        )
        freq_path = os.path.join(file_utils.get_root_directory(), "data", "local", "FORECasT", "freq", str(oligo_nr) + ".csv")
        observed_outcomes_df.to_csv(freq_path)
