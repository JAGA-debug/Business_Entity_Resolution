import pandas as pd
import re

from normalize import (
    normalize_business_name,
    normalize_address,
    normalize_country
)


# ============================================================
# CREATE BLOCKING KEYS
# ============================================================

def create_block_keys(df):

    df = df.copy()

    # --------------------------------------------------------
    # Normalize fields
    # --------------------------------------------------------

    df["name_normalized"] = (
        df["business_name"]
        .apply(normalize_business_name)
    )

    df["address_normalized"] = (
        df["business_address"]
        .apply(normalize_address)
    )

    df["country_block"] = (
        df["country"]
        .apply(normalize_country)
    )

    # ========================================================
    # NAME BLOCK 1
    # First 4 characters
    # ========================================================

    df["name_block_1"] = (
        df["name_normalized"]
        .str[:4]
    )

    # ========================================================
    # NAME BLOCK 2
    # First 4 characters without spaces
    # ========================================================

    df["name_block_2"] = (
        df["name_normalized"]
        .str.replace(" ", "", regex=False)
        .str[:4]
    )

    # ========================================================
    # NAME BLOCK 3
    # First word
    # ========================================================

    df["name_first_word"] = (
        df["name_normalized"]
        .str.split()
        .str[0]
        .fillna("")
    )

    # ========================================================
    # NAME BLOCK 4
    # First 3 characters of first word
    # ========================================================

    df["name_first_word_block"] = (
        df["name_first_word"]
        .str[:3]
    )

    # ========================================================
    # NAME BLOCK 5
    # Longest token
    # ========================================================

    def get_longest_token(name):

        if not name:
            return ""

        tokens = str(name).split()

        if not tokens:
            return ""

        return max(tokens, key=len)

    df["name_longest_token"] = (
        df["name_normalized"]
        .apply(get_longest_token)
    )

    # ========================================================
    # NAME BLOCK 6
    # First 5 characters of longest token
    # ========================================================

    df["name_longest_token_block"] = (
        df["name_longest_token"]
        .str[:5]
    )

    # ========================================================
    # ADDRESS BLOCK 7
    # First 6 characters without spaces
    # ========================================================

    df["address_block"] = (
        df["address_normalized"]
        .str.replace(" ", "", regex=False)
        .str[:6]
    )

    # ========================================================
    # ADDRESS BLOCK 8
    # First 10 characters without spaces
    # ========================================================

    df["address_block_2"] = (
        df["address_normalized"]
        .str.replace(" ", "", regex=False)
        .str[:10]
    )

    # ========================================================
    # ADDRESS BLOCK 9
    # First useful numeric token
    # ========================================================

    def get_address_token(address):

        if not address:
            return ""

        tokens = str(address).split()

        for token in tokens:

            token = token.strip()

            if re.search(r"\d", token):

                token = re.sub(
                    r"[^a-zA-Z0-9]",
                    "",
                    token
                )

                if len(token) >= 2:
                    return token.lower()

        return ""

    df["address_token_block"] = (
        df["address_normalized"]
        .apply(get_address_token)
    )

    # ========================================================
    # ADDRESS BLOCK 10
    # All numeric address tokens
    # ========================================================

    def get_address_all_tokens(address):

        if not address:
            return []

        tokens = str(address).split()

        useful_tokens = []

        for token in tokens:

            token = token.strip()

            if len(token) < 2:
                continue

            if re.search(r"\d", token):

                token = re.sub(
                    r"[^a-zA-Z0-9]",
                    "",
                    token
                )

                if len(token) >= 2:
                    useful_tokens.append(
                        token.lower()
                    )

        return list(set(useful_tokens))

    df["address_all_tokens"] = (
        df["address_normalized"]
        .apply(get_address_all_tokens)
    )

    # ========================================================
    # ADDRESS BLOCK 11
    # Meaningful alphabetic address words
    # ========================================================

    common_address_words = {
        "road",
        "street",
        "drive",
        "avenue",
        "lane",
        "city",
        "state",
        "county",
        "building",
        "block",
        "floor",
        "unit",
        "apartment",
        "house",
        "highway",
        "parkway",
        "boulevard",
        "place",
        "north",
        "south",
        "east",
        "west",
        "near",
        "opp",
        "opposite",
        "plot",
        "number",
        "india"
    }

    def get_address_word_tokens(address):

        if not address:
            return []

        tokens = str(address).split()

        useful_tokens = []

        for token in tokens:

            token = token.strip()

            # Ignore short words
            if len(token) < 5:
                continue

            # Ignore common address words
            if token in common_address_words:
                continue

            # Keep alphabetic words only
            if token.isalpha():
                useful_tokens.append(token)

        return list(set(useful_tokens))

    df["address_word_tokens"] = (
        df["address_normalized"]
        .apply(get_address_word_tokens)
    )

    return df


# ============================================================
# HELPER FUNCTION
# BUILD CANDIDATE ROWS FROM PAIRS
# ============================================================

def build_candidate_rows(
    source1_df,
    source2_df,
    pair_df
):

    if pair_df.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Keep only unique pairs
    # --------------------------------------------------------

    pair_df = pair_df[
        [
            "entity_id_s1",
            "entity_id_s2"
        ]
    ].drop_duplicates()

    # --------------------------------------------------------
    # Create lookup tables
    # --------------------------------------------------------

    source1_lookup = (
        source1_df
        .drop_duplicates("entity_id")
        .set_index(
            "entity_id",
            drop=False
        )
    )

    source2_lookup = (
        source2_df
        .drop_duplicates("entity_id")
        .set_index(
            "entity_id",
            drop=False
        )
    )

    pair_rows = []

    # --------------------------------------------------------
    # Reconstruct candidate rows
    # --------------------------------------------------------

    for pair in pair_df.itertuples(index=False):

        s1_id = pair.entity_id_s1
        s2_id = pair.entity_id_s2

        if s1_id not in source1_lookup.index:
            continue

        if s2_id not in source2_lookup.index:
            continue

        row1 = source1_lookup.loc[s1_id]
        row2 = source2_lookup.loc[s2_id]

        row = {}

        # ----------------------------------------------------
        # Source 1
        # ----------------------------------------------------

        for column in source1_df.columns:

            if column == "entity_id":

                row["entity_id_s1"] = s1_id

            elif column == "country_block":

                continue

            else:

                row[
                    column + "_s1"
                ] = row1[column]

        # ----------------------------------------------------
        # Source 2
        # ----------------------------------------------------

        for column in source2_df.columns:

            if column == "entity_id":

                row["entity_id_s2"] = s2_id

            elif column == "country_block":

                continue

            else:

                row[
                    column + "_s2"
                ] = row2[column]

        # ----------------------------------------------------
        # Common country
        # ----------------------------------------------------

        row["country_block"] = (
            row1["country_block"]
        )

        pair_rows.append(row)

    if not pair_rows:
        return pd.DataFrame()

    return pd.DataFrame(pair_rows)


# ============================================================
# GENERATE CANDIDATES
# ============================================================

def generate_candidates(
    source1_df,
    source2_df
):

    # --------------------------------------------------------
    # Create blocking keys
    # --------------------------------------------------------

    source1_df = create_block_keys(
        source1_df
    )

    source2_df = create_block_keys(
        source2_df
    )

    candidate_sets = []

    # ========================================================
    # BLOCK 1
    # Country + first 4 characters
    # ========================================================

    candidates_1 = source1_df.merge(
        source2_df,
        on=[
            "country_block",
            "name_block_1"
        ],
        suffixes=(
            "_s1",
            "_s2"
        )
    )

    if not candidates_1.empty:
        candidate_sets.append(
            candidates_1
        )

    # ========================================================
    # BLOCK 2
    # Country + first 4 chars without spaces
    # ========================================================

    candidates_2 = source1_df.merge(
        source2_df,
        on=[
            "country_block",
            "name_block_2"
        ],
        suffixes=(
            "_s1",
            "_s2"
        )
    )

    if not candidates_2.empty:
        candidate_sets.append(
            candidates_2
        )

    # ========================================================
    # BLOCK 3
    # Country + first word
    # ========================================================

    candidates_3 = source1_df.merge(
        source2_df,
        on=[
            "country_block",
            "name_first_word"
        ],
        suffixes=(
            "_s1",
            "_s2"
        )
    )

    if not candidates_3.empty:
        candidate_sets.append(
            candidates_3
        )

    # ========================================================
    # BLOCK 4
    # Country + first 3 chars of first word
    # ========================================================

    candidates_4 = source1_df.merge(
        source2_df,
        on=[
            "country_block",
            "name_first_word_block"
        ],
        suffixes=(
            "_s1",
            "_s2"
        )
    )

    if not candidates_4.empty:
        candidate_sets.append(
            candidates_4
        )

    # ========================================================
    # BLOCK 5
    # Country + longest token
    # ========================================================

    candidates_5 = source1_df.merge(
        source2_df,
        on=[
            "country_block",
            "name_longest_token"
        ],
        suffixes=(
            "_s1",
            "_s2"
        )
    )

    if not candidates_5.empty:
        candidate_sets.append(
            candidates_5
        )

    # ========================================================
    # BLOCK 6
    # Country + first 5 chars of longest token
    # ========================================================

    candidates_6 = source1_df.merge(
        source2_df,
        on=[
            "country_block",
            "name_longest_token_block"
        ],
        suffixes=(
            "_s1",
            "_s2"
        )
    )

    if not candidates_6.empty:
        candidate_sets.append(
            candidates_6
        )

    # ========================================================
    # BLOCK 7
    # Country + first 6 chars of address
    # ========================================================

    candidates_7 = source1_df.merge(
        source2_df,
        on=[
            "country_block",
            "address_block"
        ],
        suffixes=(
            "_s1",
            "_s2"
        )
    )

    if not candidates_7.empty:
        candidate_sets.append(
            candidates_7
        )

    # ========================================================
    # BLOCK 8
    # Country + first 10 chars of address
    # ========================================================

    candidates_8 = source1_df.merge(
        source2_df,
        on=[
            "country_block",
            "address_block_2"
        ],
        suffixes=(
            "_s1",
            "_s2"
        )
    )

    if not candidates_8.empty:
        candidate_sets.append(
            candidates_8
        )

    # ========================================================
    # BLOCK 9
    # Country + first numeric address token
    # ========================================================

    candidates_9 = source1_df.merge(
        source2_df,
        on=[
            "country_block",
            "address_token_block"
        ],
        suffixes=(
            "_s1",
            "_s2"
        )
    )

    if not candidates_9.empty:

        candidates_9 = candidates_9[
            candidates_9[
                "address_token_block"
            ].astype(str).str.strip() != ""
        ]

        if not candidates_9.empty:
            candidate_sets.append(
                candidates_9
            )

    # ========================================================
    # BLOCK 10
    # Country + ANY numeric address token
    # ========================================================

    s1_tokens = source1_df[
        [
            "entity_id",
            "country_block",
            "address_all_tokens"
        ]
    ].explode(
        "address_all_tokens"
    )

    s2_tokens = source2_df[
        [
            "entity_id",
            "country_block",
            "address_all_tokens"
        ]
    ].explode(
        "address_all_tokens"
    )

    s1_tokens = s1_tokens[
        s1_tokens["address_all_tokens"].notna()
    ]

    s2_tokens = s2_tokens[
        s2_tokens["address_all_tokens"].notna()
    ]

    if (
        not s1_tokens.empty
        and not s2_tokens.empty
    ):

        token_pairs = s1_tokens.merge(
            s2_tokens,
            on=[
                "country_block",
                "address_all_tokens"
            ],
            suffixes=(
                "_s1",
                "_s2"
            )
        )

        if not token_pairs.empty:

            candidates_10 = build_candidate_rows(
                source1_df,
                source2_df,
                token_pairs[
                    [
                        "entity_id_s1",
                        "entity_id_s2"
                    ]
                ]
            )

            if not candidates_10.empty:
                candidate_sets.append(
                    candidates_10
                )

    # ========================================================
    # BLOCK 11
    # Country + meaningful address word
    # ========================================================

    s1_address_words = source1_df[
        [
            "entity_id",
            "country_block",
            "address_word_tokens"
        ]
    ].explode(
        "address_word_tokens"
    )

    s2_address_words = source2_df[
        [
            "entity_id",
            "country_block",
            "address_word_tokens"
        ]
    ].explode(
        "address_word_tokens"
    )

    s1_address_words = s1_address_words[
        s1_address_words[
            "address_word_tokens"
        ].notna()
    ]

    s2_address_words = s2_address_words[
        s2_address_words[
            "address_word_tokens"
        ].notna()
    ]

    if (
        not s1_address_words.empty
        and not s2_address_words.empty
    ):

        address_word_pairs = (
            s1_address_words.merge(
                s2_address_words,
                on=[
                    "country_block",
                    "address_word_tokens"
                ],
                suffixes=(
                    "_s1",
                    "_s2"
                )
            )
        )

        if not address_word_pairs.empty:

            candidates_11 = build_candidate_rows(
                source1_df,
                source2_df,
                address_word_pairs[
                    [
                        "entity_id_s1",
                        "entity_id_s2"
                    ]
                ]
            )

            if not candidates_11.empty:
                candidate_sets.append(
                    candidates_11
                )

    # ========================================================
    # NO CANDIDATES
    # ========================================================

    if not candidate_sets:

        return pd.DataFrame()

    # ========================================================
    # COMBINE ALL BLOCKS
    # ========================================================

    candidates = pd.concat(
        candidate_sets,
        ignore_index=True
    )

    # ========================================================
    # REMOVE DUPLICATE PAIRS
    # ========================================================

    candidates = candidates.drop_duplicates(
        subset=[
            "entity_id_s1",
            "entity_id_s2"
        ]
    )

    # ========================================================
    # RETURN
    # ========================================================

    return candidates