import re
import unicodedata


def normalize_text(value):
    if value is None:
        return ""

    value = str(value)

    # Unicode normalization
    value = unicodedata.normalize("NFKC", value)

    # Lowercase
    value = value.lower()

    # Remove punctuation while preserving Unicode letters and numbers
    value = "".join(
        char if char.isalnum() or char.isspace() else " "
        for char in value
    )

    # Normalize spaces
    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalize_business_name(name):
    return normalize_text(name)


def normalize_address(address):
    return normalize_text(address)


def normalize_country(country):
    return normalize_text(country)