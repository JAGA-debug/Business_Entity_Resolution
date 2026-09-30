from rapidfuzz import fuzz
import math


GENERIC_WORDS = {
    "private",
    "limited",
    "ltd",
    "llc",
    "inc",
    "incorporated",
    "company",
    "co",
    "corporation",
    "corp",
    "group",
    "services",
    "service",
    "industries",
    "industry",
    "solutions",
    "solution",
    "enterprises",
    "enterprise",
    "international",
    "global",
    "holdings",
    "holding",
}


def clean_value(value):
    """
    Convert missing values such as NaN, None, 'nan'
    into an empty string.
    """

    if value is None:
        return ""

    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass

    value = str(value).strip()

    if value.lower() in {"nan", "none", "null"}:
        return ""

    return value


def calculate_name_similarity(name1, name2):

    name1 = clean_value(name1)
    name2 = clean_value(name2)

    if not name1 or not name2:
        return 0.0

    ratio_score = fuzz.ratio(name1, name2)
    token_score = fuzz.token_set_ratio(name1, name2)
    partial_score = fuzz.partial_ratio(name1, name2)

    return max(
        ratio_score,
        token_score,
        partial_score
    )


def calculate_address_similarity(address1, address2):

    address1 = clean_value(address1)
    address2 = clean_value(address2)

    if not address1 or not address2:
        return 0.0

    ratio_score = fuzz.ratio(address1, address2)
    token_score = fuzz.token_set_ratio(address1, address2)
    partial_score = fuzz.partial_ratio(address1, address2)

    return max(
        ratio_score,
        token_score,
        partial_score
    )


def calculate_exact_address(address1, address2):

    address1 = clean_value(address1)
    address2 = clean_value(address2)

    if not address1 or not address2:
        return False

    return address1 == address2


def calculate_address_evidence(address1, address2):

    address1 = clean_value(address1)
    address2 = clean_value(address2)

    if not address1 or not address2:
        return 0.0

    address_score = calculate_address_similarity(
        address1,
        address2
    )

    if address_score >= 95:
        return 100.0

    if address_score >= 85:
        return 90.0

    if address_score >= 75:
        return 75.0

    if address_score >= 65:
        return 60.0

    if address_score >= 50:
        return 40.0

    return address_score


def calculate_match_score(row):

    name1 = clean_value(
        row.get("name_normalized_s1", "")
    )

    name2 = clean_value(
        row.get("name_normalized_s2", "")
    )

    address1 = clean_value(
        row.get("address_normalized_s1", "")
    )

    address2 = clean_value(
        row.get("address_normalized_s2", "")
    )

    name_score = calculate_name_similarity(
        name1,
        name2
    )

    address_score = calculate_address_similarity(
        address1,
        address2
    )

    exact_address = calculate_exact_address(
        address1,
        address2
    )

    address_evidence = calculate_address_evidence(
        address1,
        address2
    )

    # Both name and address available
    if name1 and name2 and address1 and address2:

        if exact_address:
            final_score = (
                0.45 * name_score
                + 0.55 * address_evidence
            )

        else:
            final_score = (
                0.60 * name_score
                + 0.40 * address_evidence
            )

    # Only names available
    elif name1 and name2:

        final_score = name_score

    # Only addresses available
    elif address1 and address2:

        final_score = address_evidence

    else:

        final_score = 0.0

    return round(final_score, 2)


def score_candidates(candidates):

    candidates = candidates.copy()

    candidates["name_similarity"] = candidates.apply(
        lambda row: calculate_name_similarity(
            row["name_normalized_s1"],
            row["name_normalized_s2"]
        ),
        axis=1
    )

    candidates["address_similarity"] = candidates.apply(
        lambda row: calculate_address_similarity(
            row["address_normalized_s1"],
            row["address_normalized_s2"]
        ),
        axis=1
    )

    candidates["exact_address"] = candidates.apply(
        lambda row: calculate_exact_address(
            row["address_normalized_s1"],
            row["address_normalized_s2"]
        ),
        axis=1
    )

    candidates["address_evidence"] = candidates.apply(
        lambda row: calculate_address_evidence(
            row["address_normalized_s1"],
            row["address_normalized_s2"]
        ),
        axis=1
    )

    candidates["match_score"] = candidates.apply(
        calculate_match_score,
        axis=1
    )

    return candidates