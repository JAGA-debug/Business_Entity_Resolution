import pandas as pd

from blocking import generate_candidates


source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    nrows=100
)

source2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    nrows=500
)

candidates = generate_candidates(source1, source2)

print("Source 1 records:", len(source1))
print("Source 2 records:", len(source2))
print("Candidate pairs:", len(candidates))

print("\nCandidate columns:")
print(candidates.columns.tolist())

print("\nFirst 10 candidates:")
print(
    candidates[
        [
            "entity_id_s1",
            "business_name_s1",
            "entity_id_s2",
            "business_name_s2",
            "country_s1"
        ]
    ].head(10)
)
