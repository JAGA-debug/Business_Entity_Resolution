import pandas as pd
from pathlib import Path

DATASET_ROOT = Path("dataset")

source1 = pd.read_csv(
    DATASET_ROOT / "train" / "train_source1.tsv",
    sep="\t",
    nrows=5
)

source2 = pd.read_csv(
    DATASET_ROOT / "train" / "train_source2.tsv",
    sep="\t",
    nrows=5
)

source3 = pd.read_csv(
    DATASET_ROOT / "train" / "train_source3.tsv",
    sep="\t",
    nrows=5
)

ground_truth = pd.read_csv(
    DATASET_ROOT / "train" / "train_ground_truth.tsv",
    sep="\t",
    nrows=5
)

print("===== SOURCE 1 =====")
print(source1)

print("\n===== SOURCE 2 =====")
print(source2)

print("\n===== SOURCE 3 =====")
print(source3)

print("\n===== GROUND TRUTH =====")
print(ground_truth)

print("\n===== COLUMNS =====")
print("Source 1:", list(source1.columns))
print("Source 2:", list(source2.columns))
print("Source 3:", list(source3.columns))
print("Ground Truth:", list(ground_truth.columns))

print("\nDataset test successful!")