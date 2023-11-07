import os
import pandas as pd
from tqdm import tqdm
from utils import file_utils


def filter_and_center_target_sequences(settings):
    # Load raw data
    root_dir = file_utils.get_root_directory()
    targets_sequences_path = os.path.join(root_dir, "data", "target_sequences.txt")
    targets_sequences_df = pd.read_table(targets_sequences_path)

    # Length of the considered window in the target sequence (one-sided)
    target_sequence_window_extent = settings["window_length"] // 2
    accumulator = []

    for _, row in tqdm(
        targets_sequences_df.iterrows(), total=len(targets_sequences_df)
    ):
        id, _, target_sequence, scaffold, subset, _, pam_index, strand = row

        # Filter condition
        if "Explorative" in subset and strand == "FORWARD":
            # Centering the sequence
            center_index = pam_index - 3
            left_edge = center_index - target_sequence_window_extent
            right_edge = center_index + target_sequence_window_extent
            centered_target_sequence = target_sequence[left_edge:right_edge]
            accumulator.append([id, centered_target_sequence, scaffold, subset, strand])

    centered_targets_sequences_df = pd.DataFrame(
        accumulator, columns=["ID", "TargetSequence", "Scaffold", "Subset", "Strand"]
    )
    print(
        "Total number of target sequences after filtering:",
        len(centered_targets_sequences_df),
        "(out of " + str(len(targets_sequences_df)) + ")",
    )

    centered_targets_sequences_df.to_csv(
        file_utils.get_centered_target_sequences_path(settings["window_length"]),
        index=False,
    )
