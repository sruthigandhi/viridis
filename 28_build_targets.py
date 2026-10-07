"""
Viridis target construction.

Builds temporal prediction targets from county-level agricultural
land-in-farms acreage.

Target:

future_acres_change_pct =
    ((future_acres - origin_acres) / origin_acres) * 100

high_decline = 1 when future change is in the bottom 25%
within the corresponding origin year.

IMPORTANT:
The future outcome is stored here for evaluation/labeling only.
It must never enter the model feature matrix.
"""

from pathlib import Path
import pandas as pd
import numpy as np


DATA_DIR = Path("data")
OUTPUT = DATA_DIR / "targets.csv"


def clean_value(series):
    """Convert NASS Value strings to numeric values."""
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .replace({
            "(D)": np.nan,
            "(X)": np.nan,
            "(Z)": np.nan,
            "": np.nan,
            "nan": np.nan,
        })
        .astype(float)
    )


def find_land_file():
    candidates = [
        DATA_DIR / "ky_land_in_farms_raw.csv",
        DATA_DIR / "ky_farmland_raw.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    # Fall back to any CSV containing "land" or "farm"
    csvs = list(DATA_DIR.glob("*.csv"))

    for path in csvs:
        name = path.name.lower()
        if "land" in name and "income" not in name:
            return path

    raise FileNotFoundError(
        "Could not find the raw land-in-farms CSV in data/."
    )


def main():
    path = find_land_file()

    print(f"Reading: {path}")

    df = pd.read_csv(path)

    print("\nColumns:")
    print(df.columns.tolist())

    required = {"county_name", "year", "Value"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df.copy()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["acres"] = clean_value(df["Value"])

    df = df[df["year"].isin([2012, 2017, 2022])].copy()

    df["county"] = (
        df["county_name"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # If multiple NASS rows exist for the same county/year,
    # retain the first numeric acreage observation.
    land = (
        df.dropna(subset=["county", "year", "acres"])
        .sort_values(["county", "year"])
        .groupby(["county", "year"], as_index=False)["acres"]
        .first()
    )

    wide = land.pivot(
        index="county",
        columns="year",
        values="acres"
    ).reset_index()

    wide = wide.rename(columns={
        2012: "acres_2012",
        2017: "acres_2017",
        2022: "acres_2022",
    })

    rows = []

    for _, row in wide.iterrows():

        county = row["county"]

        # 2012 -> 2017
        if pd.notna(row.get("acres_2012")) and pd.notna(row.get("acres_2017")):
            change = (
                (row["acres_2017"] - row["acres_2012"])
                / row["acres_2012"]
                * 100
            )

            rows.append({
                "county": county,
                "origin_year": 2012,
                "future_year": 2017,
                "origin_acres": row["acres_2012"],
                "future_acres": row["acres_2017"],
                "future_acres_change_pct": change,
            })

        # 2017 -> 2022
        if pd.notna(row.get("acres_2017")) and pd.notna(row.get("acres_2022")):
            change = (
                (row["acres_2022"] - row["acres_2017"])
                / row["acres_2017"]
                * 100
            )

            rows.append({
                "county": county,
                "origin_year": 2017,
                "future_year": 2022,
                "origin_acres": row["acres_2017"],
                "future_acres": row["acres_2022"],
                "future_acres_change_pct": change,
            })

    targets = pd.DataFrame(rows)

    if targets.empty:
        raise ValueError("No target rows were created.")

    # Bottom quartile WITHIN each origin period.
    targets["high_decline"] = (
        targets.groupby("origin_year")[
            "future_acres_change_pct"
        ]
        .transform(lambda s: s <= s.quantile(0.25))
        .astype(int)
    )

    targets = targets.sort_values(
        ["origin_year", "county"]
    ).reset_index(drop=True)

    targets.to_csv(OUTPUT, index=False)

    print("\n=== TARGET SUMMARY ===")
    print(
        targets.groupby("origin_year")
        .agg(
            counties=("county", "count"),
            mean_change=("future_acres_change_pct", "mean"),
            median_change=("future_acres_change_pct", "median"),
            high_decline=("high_decline", "sum"),
        )
    )

    print("\nSaved:")
    print(OUTPUT)

    print("\nSample:")
    print(targets.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
