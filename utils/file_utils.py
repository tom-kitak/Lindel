import os


def get_root_directory():
    # Get the absolute path of the current file
    current_file_path = os.path.realpath(__file__)

    # Get the directory of the current file
    current_dir = os.path.dirname(current_file_path)
    # Go up in the directory hierarchy
    root_dir = os.path.dirname(current_dir)

    return root_dir


def ensure_directory_exists(directory_path):
    """Ensure that a directory exists. If it doesn't, create it."""
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Directory {directory_path} created.")
        except OSError as e:
            print(f"Error creating directory {directory_path}: {e}")


def get_centered_target_sequences_path(target_sequence_length):
    return os.path.join(
        get_root_directory(),
        "data",
        "local",
        "FORECasT",
        f"target_sequences_centered_{target_sequence_length}length.csv",
    )


def get_sorted_outcomes_folder_path():
    return os.path.join(get_root_directory(), "data", "local", "FORECasT", "sorted")


