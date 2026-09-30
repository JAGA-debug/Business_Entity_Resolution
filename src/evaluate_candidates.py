import pandas as pd

from blocking import generate_candidates


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t",
    nrows=100
)


# ============================================================
# GET SOURCE 1 IDS
# ============================================================

source1_ids = (
    ground_truth["source1_entity_id"]
    .tolist()
)


# ============================================================
# LOAD REQUIRED SOURCE 1 RECORDS
# ============================================================

source1_all = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

source1 = source1_all[
    source1_all["entity_id"].isin(source1_ids)
].copy()


# ============================================================
# COLLECT TRUE S2 MATCH IDS
# ============================================================

true_s2_ids = set()

for _, row in ground_truth.iterrows():

    matched_ids = str(
        row["matched_entity_ids"]
    ).split(",")

    for matched_id in matched_ids:

        matched_id = matched_id.strip()

        if matched_id.startswith("S2-"):

            true_s2_ids.add(
                matched_id
            )


# ============================================================
# LOAD REQUIRED SOURCE 2 RECORDS
# ============================================================

required_s2 = []

for chunk in pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    chunksize=100000
):

    matches = chunk[
        chunk["entity_id"].isin(
            true_s2_ids
        )
    ]

    if not matches.empty:

        required_s2.append(
            matches
        )

    found_count = sum(
        len(x)
        for x in required_s2
    )

    if found_count >= len(true_s2_ids):
        break


source2 = pd.concat(
    required_s2,
    ignore_index=True
)


# ============================================================
# GENERATE CANDIDATES
# ============================================================

candidates = generate_candidates(
    source1,
    source2
)


# ============================================================
# CREATE CANDIDATE PAIR SET
# ============================================================

candidate_pairs = set(
    zip(
        candidates["entity_id_s1"],
        candidates["entity_id_s2"]
    )
)


# ============================================================
# CALCULATE RECALL
# ============================================================

found = 0
total_matches = 0

missed_matches = []


for _, row in ground_truth.iterrows():

    source1_id = row[
        "source1_entity_id"
    ]

    matched_ids = [
        x.strip()
        for x in str(
            row["matched_entity_ids"]
        ).split(",")
        if x.strip()
    ]

    for matched_id in matched_ids:

        if not matched_id.startswith("S2-"):
            continue

        total_matches += 1

        pair = (
            source1_id,
            matched_id
        )

        if pair in candidate_pairs:

            found += 1

        else:

            missed_matches.append(
                pair
            )


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("===================================")
print("CANDIDATE RECALL TEST")
print("===================================")

print(
    "Source 1 records tested:",
    len(source1)
)

print(
    "Source 2 records loaded:",
    len(source2)
)

print(
    "Candidate pairs generated:",
    len(candidate_pairs)
)

print(
    "Total true S2 matches:",
    total_matches
)

print(
    "True S2 matches found:",
    found
)

if total_matches > 0:

    recall = (
        found / total_matches
    )

    print(
        "Candidate recall:",
        round(
            recall * 100,
            2
        ),
        "%"
    )

else:

    print(
        "Candidate recall: N/A"
    )


# ============================================================
# MISSED MATCHES
# ============================================================

print()
print("===================================")
print("MISSED TRUE MATCHES")
print("===================================")

print(
    "Total missed:",
    len(missed_matches)
)

if missed_matches:

    for s1_id, s2_id in missed_matches:

        print(
            s1_id,
            "->",
            s2_id
        )

print("===================================")