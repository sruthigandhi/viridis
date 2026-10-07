"""
Inspect all raw NASS CSV files before feature engineering.

This intentionally does NOT build a model.

It prints:
- files
- shapes
- columns
- years
- important NASS descriptions
- county coverage
"""

from pathlib import Path
import pandas as pd


DATA_DIR = Path("data")


def main():

    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError("No CSV files found in data/.")

    print("=" * 80)
    print("VIRIDIS RAW DATA INSPECTION")
    print("=" * 80)

    for path in files:

        print("\n" + "=" * 80)
        print(f"FILE: {path}")
        print("=" * 80)

        df = pd.read_csv(path, low_memory=False)

        print(f"Shape: {df.shape}")

        print("\nColumns:")
        for col in df.columns:
            print(f"  - {col}")

        if "year" in df.columns:
            print("\nYears:")
            print(
                sorted(
                    pd.to_numeric(
                        df["year"],
                        errors="coerce"
                    )
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

        if "county_name" in df.columns:
            print(
                f"\nUnique counties: "
                f"{df['county_name'].nunique()}"
            )

        if "short_desc" in df.columns:

            print("\nUnique short_desc values:")

            desc = (
                df["short_desc"]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .sort_values()
            )

            for value in desc.tolist():
                print(f"  {value}")

        print("\nFirst 3 rows:")
        print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
