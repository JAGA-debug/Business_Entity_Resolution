import re
import unicodedata


def normalize_text(value):
    """
    Basic text normalization.

    Steps:
    1. Handle missing values
    2. Convert to lowercase
    3. Normalize Unicode
    4. Remove extra spaces
    5. Keep letters and numbers
    """

    if value is None:
        return ""

    value = str(value)

    # Unicode normalization
    value = unicodedata.normalize("NFKC", value)

    # Lowercase
    value = value.lower()

    # Replace punctuation with spaces
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)

    # Remove extra whitespace
    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalize_business_name(name):
    """
    Normalize a business name.
    """
    return normalize_text(name)


def normalize_address(address):
    """
    Normalize a business address.
    """
    return normalize_text(address)


def normalize_country(country):
    """
    Normalize country.
    """
    return normalize_text(country)