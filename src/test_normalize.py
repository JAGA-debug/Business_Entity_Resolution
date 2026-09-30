from normalize import (
    normalize_text,
    normalize_business_name,
    normalize_address,
    normalize_country
)


names = [
    "Prime Money",
    "FOUNDATION EXCEL AGENCY PRIVATE  LIMITED",
    "Olszewski Holding Company LLC LLC",
    "Crestline Crestline Clean LP",
    "अदित्य प्रॉपर्टीज एलएलपी",
    "அரிஹந்த் Foundation Private Limited"
]

addresses = [
    "105 ELM ST, MORGANTON, NC",
    "HN 753 E-1, BHARAT NAGAR, 104/1/1 ERANDWANE"
]

print("===== BUSINESS NAME NORMALIZATION =====")

for name in names:
    print(f"Original : {name}")
    print(f"Normalized: {normalize_business_name(name)}")
    print()

print("===== ADDRESS NORMALIZATION =====")

for address in addresses:
    print(f"Original : {address}")
    print(f"Normalized: {normalize_address(address)}")
    print()

print("===== COUNTRY NORMALIZATION =====")

for country in ["US", "India", "INDIA", "US "]:
    print(f"Original : {country}")
    print(f"Normalized: {normalize_country(country)}")