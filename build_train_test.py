import os
import random

def split_files_to_txt(directory, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, output_dir="."):

    # Ensure the ratios sum to 1
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("Train, val, and test ratios must sum to 1.")

    # Get all file names in the directory
    file_names = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    random.shuffle(file_names)

    # Calculate split indices
    total_files = len(file_names)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)

    # Split the file names
    train_files = file_names[:train_end]
    val_files = file_names[train_end:val_end]
    test_files = file_names[val_end:]

    # Write to txt files
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.txt"), "w") as train_file:
        train_file.write("\n".join(train_files))
    with open(os.path.join(output_dir, "val.txt"), "w") as val_file:
        val_file.write("\n".join(val_files))
    with open(os.path.join(output_dir, "test.txt"), "w") as test_file:
        test_file.write("\n".join(test_files))


split_files_to_txt("data/images", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, output_dir="data")
