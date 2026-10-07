"""
VIRIDIS — CLEAN TEMPORAL FEATURE ENGINEERING

Builds ONLY information that would have been available at the
prediction/origin year.

Prediction periods:

    2012 -> 2017
    2017 -> 2022

Target is NOT included in the feature matrix.

NO NDVI.
NO future acreage.
NO future change.
NO target-derived variables.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from importlib.util import spec_from_file_location, module_from_spec


DATA_DIR = Path("data")
OUTPUT = DATA_DIR / "features.csv"


# ------------------------------------------------------------
# Leakage firewall
# ------------------------------------------------------------

spec = spec_from_file_location(
    "leakage_rules",
    "00_leakage_rules.py"
)

leakage_module = module_from_spec(spec)
spec.loader.exec_module(leakage_module)

assert_no_leakage = leakage_module.assert_no_leakage


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def clean_numeric(series):
    """
    Convert NASS-style numeric values into floats.
    Handles commas, $, %, and suppression markers.
    """

    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .replace({
            "(D)": np.nan,
            "(X)": np.nan,
            "(Z)": np.nan,
            "(NA)": np.nan,
            "": np.nan,
            "nan": np.nan,
            "None": np.nan,
        })
        .pipe(pd.to_numeric, errors='coerce')
    )


def normalize_county(series):
    return (
        series.astype(str)
        .str.upper()
        .str.strip()
    )


# ------------------------------------------------------------
# Load target table
# ------------------------------------------------------------

targets = pd.read_csv(DATA_DIR / "targets.csv")

targets["county"] = normalize_county(targets["county"])

print("Targets:", targets.shape)


# ------------------------------------------------------------
# LAND DATA
# ------------------------------------------------------------

land_files = [
    DATA_DIR / "ky_land_2012_raw.csv",
    DATA_DIR / "ky_land_in_farms_raw.csv",
]

land_frames = []

for path in land_files:

    if not path.exists():
        continue

    print(f"Reading land file: {path}")

    df = pd.read_csv(path, low_memory=False)

    df["county"] = normalize_county(df["county_name"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value_num"] = clean_numeric(df["Value"])

    land_frames.append(
        df[
            [
                "county",
                "year",
                "short_desc",
                "value_num",
            ]
        ]
    )

land = pd.concat(land_frames, ignore_index=True)


# ------------------------------------------------------------
# Keep only the exact land variables we want
# ------------------------------------------------------------

LAND_ACRES = "AG LAND - ACRES"

LAND_FARMS = "AG LAND - NUMBER OF OPERATIONS"

CROPLAND_ACRES = "AG LAND, CROPLAND - ACRES"


def extract_land_variable(description, output_name):

    x = land[
        land["short_desc"].eq(description)
    ].copy()

    x = (
        x.sort_values(["county", "year"])
        .drop_duplicates(
            subset=["county", "year"],
            keep="first"
        )
    )

    x = x.rename(
        columns={"value_num": output_name}
    )

    return x[
        ["county", "year", output_name]
    ]


acres = extract_land_variable(
    LAND_ACRES,
    "acres"
)

farm_count = extract_land_variable(
    LAND_FARMS,
    "farm_count"
)

cropland = extract_land_variable(
    CROPLAND_ACRES,
    "cropland_acres"
)


# Merge land variables

land_features = acres.merge(
    farm_count,
    on=["county", "year"],
    how="outer"
)

land_features = land_features.merge(
    cropland,
    on=["county", "year"],
    how="outer"
)


# Average farm size

land_features["avg_farm_size"] = (
    land_features["acres"]
    / land_features["farm_count"]
)


# ------------------------------------------------------------
# SUPPLEMENTAL EXTRA DATA
# ------------------------------------------------------------

extra_path = DATA_DIR / "extra_raw.csv"

extra = pd.read_csv(extra_path)

extra["county"] = normalize_county(extra["county"])

extra["year"] = pd.to_numeric(
    extra["year"],
    errors="coerce"
)

extra["value_num"] = clean_numeric(
    extra["value"]
)


# ------------------------------------------------------------
# Variables deliberately selected because they exist across
# the relevant prediction years.
# ------------------------------------------------------------

EXTRA_VARIABLES = {
    "netinc_per_op": "net_income_per_operation",
    "ops_loss": "operations_with_loss",
    "ops_netinc": "operations_net_income",
    "sales_per_op": "sales_per_operation",
    "govt_per_op": "government_payments_per_operation",
}


extra_parts = []

for source_name, output_name in EXTRA_VARIABLES.items():

    x = extra[
        extra["var"].eq(source_name)
    ].copy()

    if x.empty:
        print(
            f"WARNING: {source_name} not found in extra_raw.csv"
        )
        continue

    x = (
        x.sort_values(["county", "year"])
        .drop_duplicates(
            subset=["county", "year"],
            keep="first"
        )
    )

    x = x.rename(
        columns={"value_num": output_name}
    )

    extra_parts.append(
        x[
            [
                "county",
                "year",
                output_name,
            ]
        ]
    )


extra_features = None

for x in extra_parts:

    if extra_features is None:
        extra_features = x.copy()

    else:
        extra_features = extra_features.merge(
            x,
            on=["county", "year"],
            how="outer"
        )


# ------------------------------------------------------------
# Combine land + supplemental variables
# ------------------------------------------------------------

features = land_features.copy()

if extra_features is not None:

    features = features.merge(
        extra_features,
        on=["county", "year"],
        how="left"
    )


# ------------------------------------------------------------
# Create prior-period changes where valid
#
# IMPORTANT:
# We do NOT use this in the first model because 2012 does not
# have a 2007 land observation in our clean land source.
#
# It is retained as diagnostic information only.
# ------------------------------------------------------------

features = features.sort_values(
    ["county", "year"]
).reset_index(drop=True)


features["acres_chg_prior"] = (
    features.groupby("county")["acres"]
    .pct_change()
    * 100
)

features["farms_chg_prior"] = (
    features.groupby("county")["farm_count"]
    .pct_change()
    * 100
)

features["size_chg_prior"] = (
    features.groupby("county")["avg_farm_size"]
    .pct_change()
    * 100
)


# ------------------------------------------------------------
# Keep only origin years
# ------------------------------------------------------------

features = features[
    features["year"].isin([2012, 2017])
].copy()


features = features.rename(
    columns={"year": "origin_year"}
)


# ------------------------------------------------------------
# Merge with target
# ------------------------------------------------------------

features = features.merge(
    targets[
        [
            "county",
            "origin_year",
            "future_year",
            "high_decline",
        ]
    ],
    on=["county", "origin_year"],
    how="inner"
)


# ------------------------------------------------------------
# FIRST MODEL FEATURE SET
#
# We intentionally exclude prior-change variables because
# 2012 lacks a comparable 2007 land observation.
# ------------------------------------------------------------

FEATURES = [
    "acres",
    "farm_count",
    "avg_farm_size",
    "cropland_acres",
    "net_income_per_operation",
    "operations_with_loss",
    "operations_net_income",
    "sales_per_operation",
    "government_payments_per_operation",
]


# ------------------------------------------------------------
# Leakage firewall
# ------------------------------------------------------------

assert_no_leakage(FEATURES)


# Explicitly verify forbidden fields aren't features

FORBIDDEN = [
    "future_acres",
    "future_acres_change_pct",
    "high_decline",
    "future_year",
]


for forbidden in FORBIDDEN:

    assert forbidden not in FEATURES, (
        f"LEAKAGE: {forbidden} is in FEATURES"
    )


# ------------------------------------------------------------
# Basic validation
# ------------------------------------------------------------

print("\n=== FEATURE TABLE ===")

print("Shape:", features.shape)

print(
    "\nRows by origin year:"
)

print(
    features.groupby("origin_year")
    .size()
)


print("\nFeature missingness:")

print(
    features[FEATURES]
    .isna()
    .mean()
    .sort_values()
)


print("\nFeature summary:")

print(
    features[
        ["origin_year"] + FEATURES
    ].groupby("origin_year")
    .mean(numeric_only=True)
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

output_columns = [
    "county",
    "origin_year",
    "future_year",
    "high_decline",
] + FEATURES

features[
    output_columns
].to_csv(
    OUTPUT,
    index=False
)


print("\nSaved:")
print(OUTPUT)

print("\nFirst rows:")

print(
    features[
        output_columns
    ].head(10).to_string(index=False)
)


# ------------------------------------------------------------
# Final firewall check
# ------------------------------------------------------------

saved = pd.read_csv(OUTPUT)

saved_features = [
    c for c in saved.columns
    if c not in [
        "county",
        "origin_year",
        "future_year",
        "high_decline",
    ]
]

assert_no_leakage(saved_features)

assert "future_acres" not in saved.columns
assert "future_acres_change_pct" not in saved.columns

print("\n==========================================")
print("FEATURE FIREWALL PASSED")
print("==========================================")
print("No future acreage/change fields in features.")
print("No target field in model feature list.")
print("==========================================")
