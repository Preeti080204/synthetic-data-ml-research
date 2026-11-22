"""Official Adult Census Income schema from ``adult.names``."""

from __future__ import annotations

FEATURE_COLUMNS: list[str] = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]

TARGET_COLUMN = "income"
POSITIVE_CLASS = ">50K"
NEGATIVE_CLASS = "<=50K"

NUMERIC_FEATURES: list[str] = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_FEATURES: list[str] = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

METADATA_COLUMNS: list[str] = [
    "source_file",
    "source_line",
    "source_row_id",
]

# SHA-256 of the extracted UCI files shipped with this project.
# Used to confirm that loading does not rewrite the raw release.
EXPECTED_RAW_SHA256: dict[str, str] = {
    "adult.data": "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d",
    "adult.test": "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05",
    "adult.names": "c248284c0b5de30c9e1958d6cdd168a34a654758b620e68f46aefa83fc0a576a",
}

EXPECTED_ZIP_SHA256 = "7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb"

OFFICIAL_CATEGORIES: dict[str, set[str]] = {
    "workclass": {
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked",
    },
    "education": {
        "Bachelors",
        "Some-college",
        "11th",
        "HS-grad",
        "Prof-school",
        "Assoc-acdm",
        "Assoc-voc",
        "9th",
        "7th-8th",
        "12th",
        "Masters",
        "1st-4th",
        "10th",
        "Doctorate",
        "5th-6th",
        "Preschool",
    },
    "marital-status": {
        "Married-civ-spouse",
        "Divorced",
        "Never-married",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse",
    },
    "occupation": {
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv",
        "Armed-Forces",
    },
    "relationship": {
        "Wife",
        "Own-child",
        "Husband",
        "Not-in-family",
        "Other-relative",
        "Unmarried",
    },
    "race": {
        "White",
        "Asian-Pac-Islander",
        "Amer-Indian-Eskimo",
        "Other",
        "Black",
    },
    "sex": {"Female", "Male"},
    "native-country": {
        "United-States",
        "Cambodia",
        "England",
        "Puerto-Rico",
        "Canada",
        "Germany",
        "Outlying-US(Guam-USVI-etc)",
        "India",
        "Japan",
        "Greece",
        "South",
        "China",
        "Cuba",
        "Iran",
        "Honduras",
        "Philippines",
        "Italy",
        "Poland",
        "Jamaica",
        "Vietnam",
        "Mexico",
        "Portugal",
        "Ireland",
        "France",
        "Dominican-Republic",
        "Laos",
        "Ecuador",
        "Taiwan",
        "Haiti",
        "Columbia",
        "Hungary",
        "Guatemala",
        "Nicaragua",
        "Scotland",
        "Thailand",
        "Yugoslavia",
        "El-Salvador",
        "Trinadad&Tobago",
        "Peru",
        "Hong",
        "Holand-Netherlands",
    },
}

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "age": "Age in years.",
    "workclass": "Employment class of the person.",
    "fnlwgt": (
        "Census final weight. It estimates how many people in the civilian "
        "noninstitutional population the record represents. It is a survey "
        "design weight, not a personal attribute."
    ),
    "education": "Highest educational attainment.",
    "education-num": (
        "Ordinal numeric encoding of education, from preschool to doctorate."
    ),
    "marital-status": "Marital status.",
    "occupation": "Occupational category.",
    "relationship": "Relationship role of the person within the household.",
    "race": "Race category recorded in the 1994 census extract.",
    "sex": "Sex recorded in the 1994 census extract.",
    "capital-gain": "Capital gains attributed to the person.",
    "capital-loss": "Capital losses attributed to the person.",
    "hours-per-week": "Usual number of hours worked per week.",
    "native-country": "Country of origin.",
    "income": "Binary target: whether annual income exceeds $50,000.",
}

EXPECTED_ROW_COUNTS = {
    "adult.data": 32561,
    "adult.test": 16281,
    "combined": 48842,
}
