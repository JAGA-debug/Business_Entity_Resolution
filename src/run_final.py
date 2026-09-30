import os
import sqlite3
import pandas as pd

from normalize import (
    normalize_business_name,
    normalize_address,
    normalize_country
)

from scoring import (
    calculate_name_similarity,
    calculate_address_similarity,
    calculate_address_evidence,
    calculate_exact_address,
    calculate_match_score
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

TEST_DIR = os.path.join(
    DATASET_DIR,
    "test"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)

S1_FILE = os.path.join(
    TEST_DIR,
    "test_source1.tsv"
)

S2_DB = os.path.join(
    OUTPUT_DIR,
    "s2_index.db"
)

S3_DB = os.path.join(
    OUTPUT_DIR,
    "s3_index.db"
)

CANDIDATE_FILE = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs.tsv"
)

MATCH_FILE = os.path.join(
    OUTPUT_DIR,
    "matching_results.tsv"
)


# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 10_000

# SQLite-safe batch size
SQL_BATCH_SIZE = 500


# ============================================================
# CREATE BLOCK KEYS
# ============================================================

def create_block_keys(
    name,
    address,
    country
):

    keys = []

    name = str(name).strip()
    address = str(address).strip()
    country = str(country).strip()

    # --------------------------------------------------------
    # NAME BLOCKS
    # --------------------------------------------------------

    if name:

        # First 4 characters
        keys.append(
            f"C|{country}|N4|{name[:4]}"
        )

        words = name.split()

        # First word - first 3 characters
        if words:

            keys.append(
                f"C|{country}|F3|{words[0][:3]}"
            )

        # Longest word - first 5 characters
        longest = max(
            words,
            key=len,
            default=""
        )

        if longest:

            keys.append(
                f"C|{country}|L5|{longest[:5]}"
            )

    # --------------------------------------------------------
    # ADDRESS NUMERIC BLOCK
    # --------------------------------------------------------

    if address:

        for token in address.split():

            if any(
                char.isdigit()
                for char in token
            ):

                keys.append(
                    f"C|{country}|A|{token}"
                )

                break

    return list(
        set(keys)
    )


# ============================================================
# GET CANDIDATE IDs
# ============================================================

def get_candidates(
    connection,
    keys
):

    if not keys:
        return []

    result = set()

    # SQLite-safe batches
    for start in range(
        0,
        len(keys),
        SQL_BATCH_SIZE
    ):

        batch = keys[
            start:start + SQL_BATCH_SIZE
        ]

        placeholders = ",".join(
            ["?"] * len(batch)
        )

        query = f"""
            SELECT DISTINCT entity_id
            FROM blocks
            WHERE block_key IN ({placeholders})
        """

        cursor = connection.execute(
            query,
            batch
        )

        for row in cursor.fetchall():

            result.add(
                row[0]
            )

    return list(result)


# ============================================================
# GET SOURCE RECORDS
# ============================================================

def get_records(
    connection,
    entity_ids
):

    if not entity_ids:
        return {}

    result = {}

    # --------------------------------------------------------
    # IMPORTANT:
    # SQLite has a limit on SQL variables.
    # Therefore process entity IDs in batches.
    # --------------------------------------------------------

    for start in range(
        0,
        len(entity_ids),
        SQL_BATCH_SIZE
    ):

        batch = entity_ids[
            start:start + SQL_BATCH_SIZE
        ]

        placeholders = ",".join(
            ["?"] * len(batch)
        )

        query = f"""
            SELECT
                entity_id,
                business_name,
                business_address,
                country
            FROM records
            WHERE entity_id IN ({placeholders})
        """

        cursor = connection.execute(
            query,
            batch
        )

        for row in cursor.fetchall():

            result[row[0]] = {
                "entity_id": row[0],
                "business_name": row[1],
                "business_address": row[2],
                "country": row[3]
            }

    return result


# ============================================================
# SCORE ONE PAIR
# ============================================================

def score_pair(
    s1_row,
    source_row
):

    # --------------------------------------------------------
    # Normalize Source 1
    # --------------------------------------------------------

    s1_name = normalize_business_name(
        s1_row["business_name"]
    )

    s1_address = normalize_address(
        s1_row["business_address"]
    )

    # --------------------------------------------------------
    # Normalize Source 2 / Source 3
    # --------------------------------------------------------

    source_name = normalize_business_name(
        source_row["business_name"]
    )

    source_address = normalize_address(
        source_row["business_address"]
    )

    # --------------------------------------------------------
    # Temporary row for existing scoring function
    # --------------------------------------------------------

    score_row = {

        "name_normalized_s1":
            s1_name,

        "name_normalized_s2":
            source_name,

        "address_normalized_s1":
            s1_address,

        "address_normalized_s2":
            source_address
    }

    # --------------------------------------------------------
    # Individual scores
    # --------------------------------------------------------

    name_score = calculate_name_similarity(
        s1_name,
        source_name
    )

    address_score = calculate_address_similarity(
        s1_address,
        source_address
    )

    address_evidence = calculate_address_evidence(
        s1_address,
        source_address
    )

    exact_address = calculate_exact_address(
        s1_address,
        source_address
    )

    match_score = calculate_match_score(
        score_row
    )

    return (
        name_score,
        address_score,
        address_evidence,
        exact_address,
        match_score
    )


# ============================================================
# FINAL MATCH DECISION
# ============================================================

def is_match(
    name_score,
    address_score,
    address_evidence,
    exact_address,
    match_score,
    s1_address,
    source_address
):

    s1_address = str(
        s1_address
    ).strip()

    source_address = str(
        source_address
    ).strip()

    has_s1_address = bool(
        s1_address
    )

    has_source_address = bool(
        source_address
    )

    # --------------------------------------------------------
    # CASE 1:
    # Both addresses exist
    # --------------------------------------------------------

    if (
        has_s1_address
        and
        has_source_address
    ):

        # Main precision-oriented rule
        if match_score >= 70:
            return True

        # Strong address evidence
        if (
            address_evidence >= 90
            and
            name_score >= 10
        ):
            return True

        return False

    # --------------------------------------------------------
    # CASE 2:
    # One or both addresses are missing
    #
    # Be conservative because false merges are costly.
    # --------------------------------------------------------

    if name_score >= 90:
        return True

    return False


# ============================================================
# PROCESS ONE SOURCE
# ============================================================

def process_source(
    source_name,
    db_path,
    s1_chunk,
    candidate_writer,
    match_results
):

    connection = sqlite3.connect(
        db_path
    )

    try:

        for _, s1_row in s1_chunk.iterrows():

            s1_id = s1_row[
                "entity_id"
            ]

            # ------------------------------------------------
            # Normalize Source 1
            # ------------------------------------------------

            name = normalize_business_name(
                s1_row[
                    "business_name"
                ]
            )

            address = normalize_address(
                s1_row[
                    "business_address"
                ]
            )

            country = normalize_country(
                s1_row[
                    "country"
                ]
            )

            # ------------------------------------------------
            # Create blocking keys
            # ------------------------------------------------

            keys = create_block_keys(
                name,
                address,
                country
            )

            if not keys:
                continue

            # ------------------------------------------------
            # Find candidate IDs
            # ------------------------------------------------

            candidate_ids = get_candidates(
                connection,
                keys
            )

            if not candidate_ids:
                continue

            # ------------------------------------------------
            # Retrieve candidate records
            # ------------------------------------------------

            source_records = get_records(
                connection,
                candidate_ids
            )

            # ------------------------------------------------
            # Score candidates
            # ------------------------------------------------

            for candidate_id in candidate_ids:

                source_row = source_records.get(
                    candidate_id
                )

                if source_row is None:
                    continue

                (
                    name_score,
                    address_score,
                    address_evidence,
                    exact_address,
                    match_score
                ) = score_pair(
                    s1_row,
                    source_row
                )

                # ------------------------------------------------
                # Write candidate pair
                # ------------------------------------------------

                candidate_writer.write(
                    f"{s1_id}\t"
                    f"{candidate_id}\t"
                    f"{source_name}\t"
                    f"{match_score:.2f}\n"
                )

                # ------------------------------------------------
                # Decide final match
                # ------------------------------------------------

                source_address = normalize_address(
                    source_row[
                        "business_address"
                    ]
                )

                accepted = is_match(
                    name_score,
                    address_score,
                    address_evidence,
                    exact_address,
                    match_score,
                    address,
                    source_address
                )

                if accepted:

                    match_results.setdefault(
                        s1_id,
                        []
                    ).append(
                        candidate_id
                    )

    finally:

        connection.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("BUSINESS ENTITY RESOLUTION")
    print("FINAL ENTITY MATCHING")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    if not os.path.exists(
        S1_FILE
    ):

        raise FileNotFoundError(
            f"Source 1 file not found:\n{S1_FILE}"
        )

    if not os.path.exists(
        S2_DB
    ):

        raise FileNotFoundError(
            f"S2 index not found:\n{S2_DB}"
        )

    if not os.path.exists(
        S3_DB
    ):

        raise FileNotFoundError(
            f"S3 index not found:\n{S3_DB}"
        )

    # --------------------------------------------------------
    # Remove old output files
    # --------------------------------------------------------

    if os.path.exists(
        CANDIDATE_FILE
    ):

        os.remove(
            CANDIDATE_FILE
        )

    if os.path.exists(
        MATCH_FILE
    ):

        os.remove(
            MATCH_FILE
        )

    # --------------------------------------------------------
    # Store accepted matches
    # --------------------------------------------------------

    match_results = {}

    total_s1 = 0

    total_candidates = 0

    # --------------------------------------------------------
    # Open candidate output
    # --------------------------------------------------------

    with open(
        CANDIDATE_FILE,
        "w",
        encoding="utf-8",
        buffering=1024 * 1024
    ) as candidate_writer:

        candidate_writer.write(
            "source1_entity_id\t"
            "candidate_entity_id\t"
            "source\t"
            "match_score\n"
        )

        # ----------------------------------------------------
        # Read Source 1 in chunks
        # ----------------------------------------------------

        for s1_chunk in pd.read_csv(
            S1_FILE,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE
        ):

            s1_chunk = s1_chunk.fillna("")

            # ------------------------------------------------
            # Process Source 2
            # ------------------------------------------------

            process_source(
                "S2",
                S2_DB,
                s1_chunk,
                candidate_writer,
                match_results
            )

            # ------------------------------------------------
            # Process Source 3
            # ------------------------------------------------

            process_source(
                "S3",
                S3_DB,
                s1_chunk,
                candidate_writer,
                match_results
            )

            total_s1 += len(
                s1_chunk
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            print(
                f"Processed Source 1: "
                f"{total_s1:,}"
            )

    # ========================================================
    # CREATE FINAL MATCHING RESULTS
    # ========================================================

    print()
    print(
        "Creating matching_results.tsv..."
    )

    with open(
        MATCH_FILE,
        "w",
        encoding="utf-8",
        buffering=1024 * 1024
    ) as match_writer:

        match_writer.write(
            "source1_entity_id\t"
            "matched_entity_ids\n"
        )

        # ----------------------------------------------------
        # Read Source 1 again
        # ----------------------------------------------------

        for s1_chunk in pd.read_csv(
            S1_FILE,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE
        ):

            s1_chunk = s1_chunk.fillna("")

            for s1_id in s1_chunk[
                "entity_id"
            ]:

                matches = match_results.get(
                    s1_id,
                    []
                )

                # ------------------------------------------------
                # Remove duplicate matches
                # ------------------------------------------------

                matches = list(
                    dict.fromkeys(
                        matches
                    )
                )

                match_writer.write(
                    f"{s1_id}\t"
                    f"{','.join(matches)}\n"
                )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("FINAL FILES CREATED")
    print("=" * 60)

    print()
    print(
        "Candidate pairs:"
    )

    print(
        CANDIDATE_FILE
    )

    print()
    print(
        "Matching results:"
    )

    print(
        MATCH_FILE
    )

    print()
    print(
        f"Source 1 records processed: "
        f"{total_s1:,}"
    )

    print()
    print("=" * 60)
    print("FINAL MATCHING COMPLETE")
    print("=" * 60)
    print()


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    main()