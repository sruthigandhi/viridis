import numpy as np, pandas as pd
raw = pd.read_csv("data/panel_raw.csv")
w = raw.pivot_table(index="county", columns=["item", "year"], values="value")
w.columns = [f"{i}_{y}" for i, y in w.columns]

def pct(new, old):
    return ((new - old) / old * 100).replace([np.inf, -np.inf], np.nan)

def build(origin):
    prev, nxt = origin - 5, origin + 5
    d = pd.DataFrame(index=w.index)
    d["origin"] = origin
    d["acres_chg_prior"] = pct(w[f"acres_{origin}"], w[f"acres_{prev}"])
    d["farms_chg_prior"] = pct(w[f"farms_{origin}"], w[f"farms_{prev}"])
    size_o = w[f"acres_{origin}"] / w[f"farms_{origin}"]
    size_p = w[f"acres_{prev}"] / w[f"farms_{prev}"]
    d["size_chg_prior"] = pct(size_o, size_p)
    d["log_acres"] = np.log(w[f"acres_{origin}"])
    d["log_farms"] = np.log(w[f"farms_{origin}"])
    d["avg_size"] = size_o
    d["acres_chg_next"] = pct(w[f"acres_{nxt}"], w[f"acres_{origin}"])  # OUTCOME ONLY, never a feature
    return d

panel = pd.concat([build(2012), build(2017)]).dropna()
panel.to_csv("data/panel.csv")
print(panel.groupby("origin").size().rename("counties_per_origin"))
print(panel.groupby("origin")["acres_chg_next"].describe().round(2))
