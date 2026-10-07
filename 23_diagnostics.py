import pandas as pd
from scipy.stats import spearmanr

panel = pd.read_csv("data/panel.csv", index_col=0)
FEATS = ["acres_chg_prior", "farms_chg_prior", "size_chg_prior", "log_acres", "log_farms", "avg_size"]

rows = []
for origin in [2012, 2017]:
    d = panel[panel.origin == origin]
    for f in FEATS:
        rho, p = spearmanr(d[f], d.acres_chg_next)
        rows.append((origin, f, round(rho, 3), round(p, 3)))
res = pd.DataFrame(rows, columns=["origin", "feature", "rho", "p"])
print("Spearman correlation of each feature with NEXT-period acreage change")
print("(positive rho for acres_chg_prior = persistence; negative = reversal/regression to the mean)\n")
print(res.pivot(index="feature", columns="origin", values=["rho", "p"]))

print("\n--- Old features file vs fresh Census pull (2017->2022 acreage change) ---")
old = pd.read_csv("data/ky_features_final.csv", index_col=0)
old.index = old.index.astype(str).str.upper().str.strip()
new = panel[panel.origin == 2017]["acres_chg_next"].copy()
new.index = new.index.astype(str).str.upper().str.strip()
j = pd.concat([old["acreage_change_pct"].rename("old_file"), new.rename("fresh_pull")], axis=1).dropna()
print(f"Counties matched: {len(j)}")
print(j.corr().round(3))
print(j.describe().round(2))
print(j.head(8).round(2))
