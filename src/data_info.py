from pathlib import Path
import pandas as pd

DATASET_ROOT = Path("dataset")

files = [
    DATASET_ROOT / "train" / "train_source1.tsv",
    DATASET_ROOT / "train" / "train_source2.tsv",
    DATASET_ROOT / "train" / "train_source3.tsv",
    DATASET_ROOT / "train" / "train_ground_truth.tsv",
    DATASET_ROOT / "test" / "test_source1.tsv",
    DATASET_ROOT / "test" / "test_source2.tsv",
    DATASET_ROOT / "test" / "test_source3.tsv",
]

print("===== DATASET INFORMATION =====\n")

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        row_count = sum(1 for _ in f) - 1

    print(f"{file.name}: {row_count:,} rows")