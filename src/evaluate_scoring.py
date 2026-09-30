import pandas as pd
from sklearn.metrics import precision_score, recall_score, fbeta_score

from blocking import generate_candidates
from scoring import score_candidates


BASE_PATH = r"G:\AIML\BusinessEntityResolution\dataset"


# ============================================================
# 1. LOAD FIRST 100 GROUND-TRUTH RECORDS
# ============================================================

ground_truth = pd.read_csv(
    BASE_PATH + r"\train\train_ground_truth.tsv",
    sep="\t",
    nrows=100
)

ground_truth["matched_entity_ids"] = (
    ground_truth["matched_entity_ids"]
    .fillna("")
    .astype(str)
)


source1_ids = ground_truth["source1_entity_id"].tolist()


# ============================================================
# 2. LOAD REQUIRED SOURCE 1 RECORDS
# ============================================================

source1 = pd.read_csv(
    BASE_PATH + r"\train\train_source1.tsv",
    sep="\t"
)

source1 = source1[
    source1["entity_id"].isin(source1_ids)
].copy()


# ============================================================
# 3. EXTRACT TRUE SOURCE 2 IDS
# ============================================================

true_pairs = []

for _, row in ground_truth.iterrows():

    s1_id = row["source1_entity_id"]

    matched_ids = [
        x.strip()
        for x in row["matched_entity_ids"].split(",")
        if x.strip()
    ]

    for s2_id in matched_ids:

        if s2_id.startswith("S2-"):

            true_pairs.append(
                (s1_id, s2_id)
            )


true_pair_set = set(true_pairs)

true_s2_ids = sorted(
    set(
        s2_id
        for _, s2_id in true_pairs
    )
)


# ============================================================
# 4. LOAD ONLY REQUIRED SOURCE 2 RECORDS
# ============================================================

source2 = pd.read_csv(
    BASE_PATH + r"\train\train_source2.tsv",
    sep="\t"
)

source2 = source2[
    source2["entity_id"].isin(true_s2_ids)
].copy()


# ============================================================
# 5. GENERATE CANDIDATES
# ============================================================

candidates = generate_candidates(
    source1,
    source2
)

print()
print("===================================")
print("SCORING EVALUATION")
print("===================================")

print(
    "Candidate pairs:",
    len(candidates)
)


# ============================================================
# 6. SCORE CANDIDATES
# ============================================================

candidates = score_candidates(
    candidates
)


# ============================================================
# 7. LABEL TRUE / FALSE MATCHES
# ============================================================

candidates["is_true_match"] = candidates.apply(
    lambda row:
        (
            row["entity_id_s1"],
            row["entity_id_s2"]
        ) in true_pair_set,
    axis=1
)


# ============================================================
# 8. SCORE DISTRIBUTION
# ============================================================

true_scores = candidates.loc[
    candidates["is_true_match"],
    "match_score"
]

false_scores = candidates.loc[
    ~candidates["is_true_match"],
    "match_score"
]


print()
print("-----------------------------------")
print("TRUE MATCH SCORE DISTRIBUTION")
print("-----------------------------------")

print("Count:", len(true_scores))
print("Minimum:", round(true_scores.min(), 2))
print("Average:", round(true_scores.mean(), 2))
print("Maximum:", round(true_scores.max(), 2))


print()
print("-----------------------------------")
print("NON-MATCH SCORE DISTRIBUTION")
print("-----------------------------------")

print("Count:", len(false_scores))
print("Minimum:", round(false_scores.min(), 2))
print("Average:", round(false_scores.mean(), 2))
print("Maximum:", round(false_scores.max(), 2))


# ============================================================
# 9. F0.5 THRESHOLD ANALYSIS
# ============================================================

thresholds = [
    40,
    45,
    50,
    55,
    60,
    65,
    70,
    75,
    80,
    85,
    90,
    95
]


print()
print("===================================")
print("F0.5 THRESHOLD ANALYSIS")
print("===================================")


for threshold in thresholds:

    predictions = (
        candidates["match_score"]
        >= threshold
    )

    actual = candidates[
        "is_true_match"
    ]

    precision = precision_score(
        actual,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        actual,
        predictions,
        zero_division=0
    )

    f05 = fbeta_score(
        actual,
        predictions,
        beta=0.5,
        zero_division=0
    )

    selected = predictions.sum()

    true_selected = (
        candidates.loc[
            predictions,
            "is_true_match"
        ]
        .sum()
    )

    false_selected = (
        selected
        - true_selected
    )

    print(
        f"Threshold {threshold:>3} | "
        f"Selected: {selected:>4} | "
        f"True: {true_selected:>4} | "
        f"False: {false_selected:>4} | "
        f"Precision: {precision * 100:>6.2f}% | "
        f"Recall: {recall * 100:>6.2f}% | "
        f"F0.5: {f05 * 100:>6.2f}%"
    )


# ============================================================
# 10. TOP FALSE MATCHES
# ============================================================

print()
print("===================================")
print("TOP NON-MATCHES")
print("===================================")

top_false = candidates[
    ~candidates["is_true_match"]
].sort_values(
    "match_score",
    ascending=False
).head(20)

print(
    top_false[
        [
            "entity_id_s1",
            "business_name_s1",
            "entity_id_s2",
            "business_name_s2",
            "name_similarity",
            "address_similarity",
            "match_score"
        ]
    ].to_string(index=False)
)


# ============================================================
# 11. LOWEST TRUE MATCHES
# ============================================================

print()
print("===================================")
print("LOWEST SCORING TRUE MATCHES")
print("===================================")

lowest_true = candidates[
    candidates["is_true_match"]
].sort_values(
    "match_score"
).head(20)

print(
    lowest_true[
        [
            "entity_id_s1",
            "business_name_s1",
            "entity_id_s2",
            "business_name_s2",
            "name_similarity",
            "address_similarity",
            "match_score"
        ]
    ].to_string(index=False)
)


# ============================================================
# 12. DONE
# ============================================================

print()
print("===================================")
print("DONE")
print("===================================")