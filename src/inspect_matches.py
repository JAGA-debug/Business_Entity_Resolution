import pandas as pd


# Load first 100 ground-truth records
ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t",
    nrows=100
)

# Get the actual S1 IDs from ground truth
source1_ids = ground_truth["source1_entity_id"].tolist()

# Load Source 1 and find those exact IDs
source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

source1 = source1[
    source1["entity_id"].isin(source1_ids)
]

# Load Source 2
source2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t"
)

# Fast lookup
source2_lookup = source2.set_index("entity_id")


print("===================================")
print("REAL GROUND-TRUTH MATCHES")
print("===================================")

shown = 0

for _, gt_row in ground_truth.iterrows():

    source1_id = gt_row["source1_entity_id"]

    s1_match = source1[
        source1["entity_id"] == source1_id
    ]

    if s1_match.empty:
        continue

    s1_row = s1_match.iloc[0]

    matched_ids = [
        x.strip()
        for x in str(gt_row["matched_entity_ids"]).split(",")
        if x.strip()
    ]

    for matched_id in matched_ids:

        if not matched_id.startswith("S2-"):
            continue

        if matched_id not in source2_lookup.index:
            continue

        s2_row = source2_lookup.loc[matched_id]

        print("\n-----------------------------------")

        print("Source 1 ID:", source1_id)
        print("Source 1 Name:", s1_row["business_name"])
        print("Source 1 Address:", s1_row["business_address"])
        print("Source 1 Country:", s1_row["country"])

        print("\nSource 2 ID:", matched_id)
        print("Source 2 Name:", s2_row["business_name"])
        print("Source 2 Address:", s2_row["business_address"])
        print("Source 2 Country:", s2_row["country"])

        shown += 1

        if shown >= 10:
            break

    if shown >= 10:
        break


print("\n===================================")
print("Matches displayed:", shown)
print("===================================")