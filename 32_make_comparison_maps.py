import pickle
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# =========================================================
# 1. LOAD DATA
# =========================================================

panel = pd.read_csv("data/panel.csv")

# 2017 is the prediction origin.
# The actual outcome is 2017 -> 2022.
test = panel[panel["origin"] == 2017].copy()

# =========================================================
# 2. LOAD THE TRAINED BASELINE MODEL
# =========================================================

with open("model_v4_temporal.pkl", "rb") as f:
    saved = pickle.load(f)

models = saved["models"]

baseline_name = (
    "Persistence baseline (last period's acreage change only)"
)

cols, model = models[baseline_name]

# =========================================================
# 3. PREDICT 2017 -> 2022 RISK
# =========================================================

test["predicted_score"] = model.predict_proba(
    test[cols]
)[:, 1]

# Convert predicted scores to a 0-100 percentile.
# 100 = counties the model ranks at highest risk.
test["predicted_risk"] = (
    test["predicted_score"].rank(pct=True) * 100
)

# =========================================================
# 4. CONVERT OBSERVED DECLINE TO THE SAME 0-100 SCALE
# =========================================================

# acres_chg_next is the actual percentage change in
# agricultural acreage from 2017 -> 2022.
#
# More negative = greater decline = greater risk.
#
# Rank the actual changes so that the largest declines
# receive the highest risk percentile.

test["observed_risk"] = (
    100 - test["acres_chg_next"].rank(pct=True) * 100
)

# =========================================================
# 5. SAVE THE COMPARISON DATA
# =========================================================

comparison = test[
    [
        "county",
        "predicted_score",
        "predicted_risk",
        "acres_chg_next",
        "observed_risk",
    ]
].copy()

comparison.to_csv(
    "data/2017_prediction_vs_observed.csv",
    index=False
)

# =========================================================
# 6. LOAD KENTUCKY COUNTY BOUNDARIES
# =========================================================

url = (
    "https://www2.census.gov/geo/tiger/"
    "GENZ2024/shp/cb_2024_us_county_500k.zip"
)

counties = gpd.read_file(url)

# Kentucky FIPS = 21
ky = counties[counties["STATEFP"] == "21"].copy()

# =========================================================
# 7. CLEAN COUNTY NAMES
# =========================================================

def clean_name(x):
    return (
        str(x)
        .upper()
        .replace(" COUNTY", "")
        .replace(" PARISH", "")
        .replace(" CITY AND BOROUGH", "")
        .strip()
    )

ky["county_clean"] = ky["NAME"].apply(clean_name)
comparison["county_clean"] = comparison["county"].apply(clean_name)

# =========================================================
# 8. JOIN DATA TO COUNTY MAP
# =========================================================

map_data = ky.merge(
    comparison,
    on="county_clean",
    how="left"
)

print("\nMap coverage")
print("-------------------------")
print(f"Kentucky counties: {len(ky)}")
print(
    f"Predicted risk: "
    f"{map_data['predicted_risk'].notna().sum()}"
)
print(
    f"Observed risk: "
    f"{map_data['observed_risk'].notna().sum()}"
)

missing = map_data.loc[
    map_data["predicted_risk"].isna(),
    "NAME"
].tolist()

if missing:
    print("\nCounties without complete temporal data:")
    print(", ".join(missing))

# =========================================================
# 9. COMMON COLOR SCALE
# =========================================================

# BOTH maps use exactly the same 0-100 scale.
norm = Normalize(vmin=0, vmax=100)

# Same color palette for both maps.
cmap = "YlGn"

# =========================================================
# 10. CREATE LARGE SIDE-BY-SIDE MAPS
# =========================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 8.5),
    gridspec_kw={
        "wspace": 0.03
    }
)

# ---------------------------------------------------------
# LEFT: PREDICTED RISK
# ---------------------------------------------------------

map_data.plot(
    column="predicted_risk",
    cmap=cmap,
    norm=norm,
    linewidth=0.35,
    edgecolor="white",
    ax=axes[0],
    missing_kwds={
        "color": "lightgrey",
        "edgecolor": "white"
    }
)

axes[0].set_title(
    "Predicted Risk",
    fontsize=17,
    fontweight="bold",
    pad=8
)

axes[0].axis("off")

# ---------------------------------------------------------
# RIGHT: OBSERVED RISK
# ---------------------------------------------------------

map_data.plot(
    column="observed_risk",
    cmap=cmap,
    norm=norm,
    linewidth=0.35,
    edgecolor="white",
    ax=axes[1],
    missing_kwds={
        "color": "lightgrey",
        "edgecolor": "white"
    }
)

axes[1].set_title(
    "Observed Risk",
    fontsize=17,
    fontweight="bold",
    pad=8
)

axes[1].axis("off")

# =========================================================
# 11. SHARED COLORBAR
# =========================================================

sm = ScalarMappable(
    norm=norm,
    cmap=cmap
)

sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=axes,
    orientation="horizontal",
    fraction=0.045,
    pad=0.025,
    aspect=55
)

cbar.set_label(
    "Risk percentile",
    fontsize=13,
    labelpad=6
)

cbar.set_ticks([0, 25, 50, 75, 100])

cbar.set_ticklabels([
    "0  Lowest",
    "25",
    "50",
    "75",
    "100  Highest"
])

# =========================================================
# 12. COMPACT OVERALL TITLE
# =========================================================

fig.suptitle(
    "Predicted vs. Observed Agricultural-Land Decline Risk",
    fontsize=21,
    fontweight="bold",
    y=0.97
)

fig.text(
    0.5,
    0.935,
    "Kentucky counties, 2017–2022",
    ha="center",
    fontsize=12
)

# Small explanation below the colorbar
fig.text(
    0.5,
    0.015,
    "Higher percentile = greater predicted or observed decline risk",
    ha="center",
    fontsize=10
)

# =========================================================
# 13. FINAL LAYOUT
# =========================================================

plt.subplots_adjust(
    left=0.015,
    right=0.985,
    top=0.88,
    bottom=0.12,
    wspace=0.02
)

# =========================================================
# 14. SAVE
# =========================================================

plt.savefig(
    "figures/kentucky_predicted_vs_observed.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "figures/kentucky_predicted_vs_observed.pdf",
    bbox_inches="tight"
)

plt.close()

print("\nSaved:")
print("figures/kentucky_predicted_vs_observed.png")
print("figures/kentucky_predicted_vs_observed.pdf")
