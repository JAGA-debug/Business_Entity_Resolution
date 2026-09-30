import pandas as pd

from blocking import generate_candidates
from scoring import score_candidates


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


# Generate candidate pairs
candidates = generate_candidates(
    source1,
    source2
)


# Calculate similarity scores
scored_candidates = score_candidates(
    candidates
)


print("Candidate pairs:", len(scored_candidates))

print("\nScored candidates:")

print(
    scored_candidates[
        [
            "entity_id_s1",
            "business_name_s1",
            "entity_id_s2",
            "business_name_s2",
            "name_similarity",
            "address_similarity",
            "match_score"
        ]
    ].head(20).to_string(index=False)
)