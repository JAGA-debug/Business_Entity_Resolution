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

    # Keep letters, numbers, combining marks and spaces.
    # This is important for Indian scripts.
    cleaned = []

    for char in value:

        category = unicodedata.category(char)

        if (
            char.isspace()
            or category.startswith("L")
            or category.startswith("N")
            or category.startswith("M")
        ):
            cleaned.append(char)
        else:
            cleaned.append(" ")

    value = "".join(cleaned)

    # Normalize whitespace
    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalize_business_name(name):
    return normalize_text(name)


def normalize_address(address):
    return normalize_text(address)


def normalize_country(country):
    return normalize_text(country)