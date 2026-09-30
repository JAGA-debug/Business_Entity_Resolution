import pandas as pd
from pathlib import Path

DATASET_ROOT = Path("dataset")

source1 = pd.read_csv(
    DATASET_ROOT / "train" / "train_source1.tsv",
    sep="\t",
    nrows=20
)

source2 = pd.read_csv(
    DATASET_ROOT / "train" / "train_source2.tsv",
    sep="\t",
    nrows=20
)

source3 = pd.read_csv(
    DATASET_ROOT / "train" / "train_source3.tsv",
    sep="\t",
    nrows=20
)

print("\n========== SOURCE 1 ==========\n")
print(source1[
    ["entity_id", "business_name", "business_address", "country"]
].to_string(index=False))

print("\n========== SOURCE 2 ==========\n")
print(source2[
    ["entity_id", "business_name", "business_address", "country"]
].to_string(index=False))

print("\n========== SOURCE 3 ==========\n")
print(source3[
    ["entity_id", "business_name", "business_address", "country"]
].to_string(index=False))