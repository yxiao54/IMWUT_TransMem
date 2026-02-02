# ============================================================
# FULL END-TO-END RQ1 PIPELINE 
# From data/windows.parquet to stress / craving / resilience PCA results
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ===================== CONFIG =====================
PARQUET_PATH = "data/windows.parquet"
OUT_DIR = "rq1_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


MIN_WINDOWS = 3
MIN_STD = 1e-6
Z_CLIP = 5.0

# ===================== UTILITIES =====================
def majority_vote(x):
    x = x.dropna().astype(int)
    if len(x) == 0:
        return np.nan
    return int(x.sum() >= len(x) / 2)

def global_zscore(df, cols):
    df = df.copy()
    for c in cols:
        mu, sd = df[c].mean(), df[c].std()
        df[c] = ((df[c] - mu) / sd).clip(-Z_CLIP, Z_CLIP) if sd > 0 else 0.0
    return df

def gee_fit(df, y, x):
    """
    Generalized GEE fit that handles:
      - numeric predictors like 'stress' or 'craving' (use param name x)
      - categorical predictor 'Resilience' (use C(Resilience)[T.Low] key)
    Returns dict or None.
    """
    # prepare subframe
    cols_needed = ["user", "Group", "n_windows", y]
    # x may be categorical name 'Resilience' or numeric column
    cols_needed.append(x)
    sub = df[cols_needed].dropna()
    if len(sub) < 10 or sub[y].std() < MIN_STD:
        return None

    # build formula
    if x == "Resilience":
        formula = f"{y} ~ C(Resilience) + C(Group) + n_windows"
    else:
        formula = f"{y} ~ {x} + C(Group) + n_windows"

    try:
        model = smf.gee(
            formula,
            groups="user",
            data=sub,
            family=sm.families.Gaussian(),
            cov_struct=sm.cov_struct.Exchangeable()
        )
        res = model.fit(cov_type="robust")

        # determine parameter key to extract
        if x == "Resilience":
            key = "C(Resilience)[T.Low]"
            if key not in res.params.index:
                # robust fallback: find any param that starts with 'C(Resilience)'
                candidates = [k for k in res.params.index if k.startswith("C(Resilience)")]
                key = candidates[0] if candidates else None
        else:
            key = x
            if key not in res.params.index:
                # fallback: sometimes factor encoded differently; try to find exact match ignoring dtype
                candidates = [k for k in res.params.index if k.endswith(f"{x}")]
                key = candidates[0] if candidates else None

        if key is None or key not in res.params.index:
            return None

        return {
            "coef": res.params[key],
            "se": res.bse[key],
            "p": res.pvalues[key],
            "ci_low": res.conf_int().loc[key][0],
            "ci_high": res.conf_int().loc[key][1],
            "n_obs": len(sub)
        }
    except Exception as e:
        # on any failure, return None (keeps pipeline robust)
        return None

# ===================== MAIN =====================
def main():
    print("Loading parquet...")
    win = pd.read_parquet(PARQUET_PATH)
    win["user"] = win["user"].astype(str)

    stress_col = "stress" if "stress" in win.columns else None
    craving_col = "Craving_bin" if "Craving_bin" in win.columns else None

    meta_cols = {"user","task","window_start_sec","segment_start_utc","ema_time_utc"}
    if stress_col: meta_cols.add(stress_col)
    if craving_col: meta_cols.add(craving_col)

    feature_cols = [c for c in win.columns if c not in meta_cols]
    win = global_zscore(win, feature_cols)

    # ---------- task-level aggregation ----------
    agg = {f+"_mean": (f,"mean") for f in feature_cols}
    agg["n_windows"] = ("window_start_sec","count")
    task = win.groupby(["user","task"], as_index=False).agg(**agg)
    task = task[task["n_windows"] >= MIN_WINDOWS]

    # ---------- baseline delta ----------
    base = task[task["task"] == BASELINE_TASK][["user"] + [f+"_mean" for f in feature_cols]]
    base = base.rename(columns={c: c+"_baseline" for c in base.columns if c != "user"})
    task = task.merge(base, on="user", how="left")
    for f in feature_cols:
        task[f+"_delta"] = task[f+"_mean"] - task[f+"_mean_baseline"]

    # ---------- labels ----------
    task["Group"] = task["user"].map(lambda u: "OUD" if u in OUD_USERS else "Control")
    task["Resilience"] = task["user"].map(lambda u: "High" if u in resilient_high else "Low")

    # ---------- stress / craving ----------
    if stress_col:
        s = win.groupby(["user","task"])[stress_col].apply(majority_vote).reset_index(name="stress")
        task = task.merge(s, on=["user","task"], how="left")
    else:
        task["stress"] = task["task"].isin(STRESS_TASKS).astype(int)

    if craving_col:
        c = win.groupby(["user","task"])[craving_col].apply(majority_vote).reset_index(name="craving")
        task = task.merge(c, on=["user","task"], how="left")
    else:
        task["craving"] = np.nan

    task["Group"] = pd.Categorical(task["Group"], ["Control","OUD"])
    task["Resilience"] = pd.Categorical(task["Resilience"], ["High","Low"])

    task.to_csv(os.path.join(OUT_DIR,"task_level_from_raw.csv"), index=False)

    # ===================== PCA-ONLY SYSTEM-LEVEL (ANALYSIS C) =====================
    SYSTEMS = {
        "Cardiovascular":[c+"_delta" for c in feature_cols if c.startswith(("BVP_","HR_HR_","HR_l2_","HRV_"))],
        "Electrodermal":[c+"_delta" for c in feature_cols if c.startswith(("Tonic_","Phasic_","EDA_"))],
        "Movement":[c+"_delta" for c in feature_cols if c.startswith("ACC_")],
        "Thermoregulation":[c+"_delta" for c in feature_cols if c.startswith("TEMP_")]
    }

    FACTORS = ["stress","craving","Resilience"]
    rows = []

    print("\n===== PCA SYSTEM-LEVEL RESULTS =====")

    for x in FACTORS:
        for sys, cols in SYSTEMS.items():
            cols = [c for c in cols if c in task.columns and task[c].std() > MIN_STD]
            if len(cols) < 2:
                continue

            Z = (task[cols] - task[cols].mean()) / (task[cols].std() + 1e-8)
            Z = Z.fillna(0.0)

            pca = PCA(n_components=1)
            task[f"{sys}_pc1"] = pca.fit_transform(Z.values).ravel()

            out = gee_fit(task, f"{sys}_pc1", x)
            if out:
                rows.append({
                    "factor": x,
                    "system": sys,
                    "explained_var": pca.explained_variance_ratio_[0],
                    "n_features": len(cols),
                    **out
                })
                print(
                    f"{x:10s} | {sys:16s} | "
                    f"beta={out['coef']:.3f}, p={out['p']:.3g}, "
                    f"PC1 var={pca.explained_variance_ratio_[0]:.3f}"
                )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(OUT_DIR,"rq1_system_pca_all_factors.csv"), index=False)
    print("\nSaved rq1_system_pca_all_factors.csv")
    print("PIPELINE COMPLETE.")

if __name__ == "__main__":
    main()
