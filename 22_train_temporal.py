import pickle, numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

panel = pd.read_csv("data/panel.csv", index_col=0)
train, test = panel[panel.origin == 2012], panel[panel.origin == 2017]
FEATS = ["acres_chg_prior", "farms_chg_prior", "size_chg_prior", "log_acres", "log_farms", "avg_size"]

thr = train.acres_chg_next.quantile(0.25)   # threshold from TRAINING outcomes only
y_tr = (train.acres_chg_next <= thr).astype(int)
y_te = (test.acres_chg_next <= thr).astype(int)
print(f"High-risk = next-5yr acreage change <= {thr:.2f}%")
print(f"Base rate: train {y_tr.mean():.2f} | test {y_te.mean():.2f} (n_train={len(train)}, n_test={len(test)})\n")

models = {
    "Persistence baseline (last period's acreage change only)": (["acres_chg_prior"],
        make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))),
    "Logistic regression (6 features)": (FEATS,
        make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))),
    "Random forest (6 features)": (FEATS,
        RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=5,
                               class_weight="balanced_subsample", random_state=42)),
}

def boot_auc(y, p, n=1000, seed=0):
    rng, aucs = np.random.default_rng(seed), []
    y, p = np.asarray(y), np.asarray(p)
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            aucs.append(roc_auc_score(y[i], p[i]))
    return np.percentile(aucs, [2.5, 97.5])

k = int(round(len(test) * 0.25))
fitted = {}
for name, (cols, m) in models.items():
    m.fit(train[cols], y_tr)
    p = m.predict_proba(test[cols])[:, 1]
    top = np.argsort(-p)[:k]
    lo, hi = boot_auc(y_te, p)
    print(name)
    print(f"  AUC {roc_auc_score(y_te, p):.3f} (95% CI {lo:.2f}-{hi:.2f}) | avg precision {average_precision_score(y_te, p):.3f} (chance = {y_te.mean():.2f})")
    print(f"  Precision in top {k} flagged counties: {y_te.values[top].mean():.2f}\n")
    fitted[name] = (cols, m)

with open("model_v4_temporal.pkl", "wb") as f:
    pickle.dump({"models": fitted, "threshold": thr, "features": FEATS}, f)
print("Saved model_v4_temporal.pkl")
