import pandas as pd

from blocking import generate_candidates
from scoring import score_candidates


# ==========================================
# SETTINGS
# ==========================================

SAMPLE_SIZE = 1000

TRAIN_PATH = "dataset/train/"


# ==========================================
# LOAD GROUND TRUTH
# ==========================================

print("Loading ground truth...")

ground_truth = pd.read_csv(
    TRAIN_PATH + "train_ground_truth.tsv",
    sep="\t"
)

ground_truth = ground_truth.head(SAMPLE_SIZE)

print("Ground truth records:", len(ground_truth))


# ==========================================
# LOAD SOURCE 1
# ==========================================

print("Loading Source 1...")

source1_ids = ground_truth["source1_entity_id"].tolist()

source1 = pd.read_csv(
    TRAIN_PATH + "train_source1.tsv",
    sep="\t"
)

source1 = source1[
    source1["entity_id"].isin(source1_ids)
].copy()

print("Source 1 records loaded:", len(source1))


# ==========================================
# GET TRUE SOURCE 2 IDs
# ==========================================

true_pairs = []

for _, row in ground_truth.iterrows():

    s1_id = row["source1_entity_id"]

    matched_ids = str(
        row["matched_entity_ids"]
    ).split(",")

    for matched_id in matched_ids:

        matched_id = matched_id.strip()

        if matched_id.startswith("S2-"):

            true_pairs.append(
                (s1_id, matched_id)
            )


true_pairs_df = pd.DataFrame(
    true_pairs,
    columns=["s1_id", "s2_id"]
)

true_s2_ids = true_pairs_df["s2_id"].unique().tolist()

print("True Source 2 matches:", len(true_pairs_df))
print("Required Source 2 records:", len(true_s2_ids))


# ==========================================
# LOAD SOURCE 2
# ==========================================

print("Loading Source 2...")

source2 = pd.read_csv(
    TRAIN_PATH + "train_source2.tsv",
    sep="\t"
)

source2 = source2[
    source2["entity_id"].isin(true_s2_ids)
].copy()

print("Source 2 records loaded:", len(source2))


# ==========================================
# GENERATE CANDIDATES
# ==========================================

print()
print("Generating candidates...")

candidates = generate_candidates(
    source1,
    source2
)

print("Candidate pairs:", len(candidates))


# ==========================================
# SCORE CANDIDATES
# ==========================================

print()
print("Scoring candidates...")

candidates = score_candidates(candidates)


# ==========================================
# MARK TRUE MATCHES
# ==========================================

true_pair_set = set(
    zip(
        true_pairs_df["s1_id"],
        true_pairs_df["s2_id"]
    )
)

candidates["is_true_match"] = candidates.apply(
    lambda row: (
        row["entity_id_s1"],
        row["entity_id_s2"]
    ) in true_pair_set,
    axis=1
)


# ==========================================
# RULE 5
# ==========================================

candidates["rule5_selected"] = (

    (candidates["match_score"] >= 70)

    |

    (
        (candidates["address_evidence"] >= 90)
        &
        (candidates["name_similarity"] >= 10)
    )
)


selected = candidates[
    candidates["rule5_selected"]
]

true_selected = selected[
    selected["is_true_match"]
]

false_selected = selected[
    ~selected["is_true_match"]
]


# ==========================================
# METRICS
# ==========================================

selected_count = len(selected)

true_count = len(true_selected)

false_count = len(false_selected)

total_true = len(true_pair_set)


if selected_count > 0:

    precision = true_count / selected_count

else:

    precision = 0


if total_true > 0:

    recall = true_count / total_true

else:

    recall = 0


if precision + recall > 0:

    f05 = (
        1.25
        * precision
        * recall
        /
        (0.25 * precision + recall)
    )

else:

    f05 = 0


# ==========================================
# RESULT
# ==========================================

print()
print("===================================")
print("RULE 5 EVALUATION - 1000 RECORDS")
print("===================================")

print("Candidate pairs:", len(candidates))

print()
print("-----------------------------------")
print("Rule 5")
print("-----------------------------------")

print("Selected:", selected_count)

print("True:", true_count)

print("False:", false_count)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall: {recall * 100:.2f}%"
)

print(
    f"F0.5: {f05 * 100:.2f}%"
)

print()
print("===================================")
print("DONE")
print("===================================")
print()
print("===================================")
print("TOP FALSE MATCHES")
print("===================================")

false_matches = selected[
    ~selected["is_true_match"]
].copy()

false_matches = false_matches.sort_values(
    "match_score",
    ascending=False
)

print(
    false_matches[
        [
            "entity_id_s1",
            "entity_id_s2",
            "name_normalized_s1",
            "name_normalized_s2",
            "address_normalized_s1",
            "address_normalized_s2",
            "name_similarity",
            "address_similarity",
            "address_evidence",
            "match_score"
        ]
    ].head(20).to_string(index=False)
)

print()
print("===================================")
print("DONE")
print("===================================")