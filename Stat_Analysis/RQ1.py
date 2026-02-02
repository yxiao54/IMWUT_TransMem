# rq1_final_global_zscore.py
# ============================================================
# RQ1 FINAL SCRIPT (STATISTICAL / MECHANISTIC ANALYSIS)
# Global z-score → task-level aggregation → jelly baseline delta
# ============================================================

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
PARQUET_PATH = "data/windows.parquet"   # 
OUT_PREFIX = "rq1_final"


MIN_WINDOWS = 3
Z_CLIP = 5.0          # winsorization after z-score
MIN_STD = 1e-6        # minimum variance for modeling

DROP_FEATURES = {
    'BVP_BVP_entropy','ACC_x_entropy','ACC_y_entropy','ACC_z_entropy',
    'Phasic_phasic_entropy','HRV_ULF','HRV_VLF','HRV_LF','HRV_LFHF',
    'HRV_LFn','HRV_SDANN1','HRV_SDNNI1','HRV_SDANN2','HRV_SDNNI2',
    'HRV_SDANN5','HRV_SDNNI5'
}




# ============================================================
# LOAD & FEATURE SELECTION
# ============================================================
def load_windows():
    df = pd.read_parquet(PARQUET_PATH)
    df["user"] = df["user"].astype(str)
    df = df[df["task"].isin(TASKS_KEEP)].copy()
    return df

def get_feature_cols(df):
    meta = {
        "user","task","stress","craving",
        "window_start_sec","rr_count","hrv_rr_valid",
        "segment_start_utc","ema_time_utc"
    }
    return [c for c in df.columns if c not in meta and c not in DROP_FEATURES]

# ============================================================
# GLOBAL Z-SCORE (RQ1 ONLY)
# ============================================================
def global_zscore(df, feature_cols):
    df = df.copy()
    for f in feature_cols:
        mu = df[f].mean()
        sd = df[f].std()
        if sd > 0:
            z = (df[f] - mu) / sd
            df[f] = z.clip(-Z_CLIP, Z_CLIP)
        else:
            df[f] = 0.0
    return df

# ============================================================
# TASK-LEVEL AGGREGATION
# ============================================================
def aggregate_task_level(df, feature_cols):
    agg = {}
    for f in feature_cols:
        agg[f+"_mean"] = (f, "mean")
        agg[f+"_sd"]   = (f, "std")

    task_df = (
        df.groupby(["user","task"])
          .agg(**agg, n_windows=("window_start_sec","count"))
          .reset_index()
    )
    task_df = task_df[task_df["n_windows"] >= MIN_WINDOWS]
    return task_df

def compute_baseline_delta(task_df, feature_cols):
    mean_cols = [f+"_mean" for f in feature_cols]
    baseline = (
        task_df[task_df["task"] == BASELINE_TASK]
        [["user"] + mean_cols]
        .rename(columns={c: c+"_baseline" for c in mean_cols})
    )
    df = task_df.merge(baseline, on="user", how="left")
    for f in feature_cols:
        df[f+"_delta"] = df[f+"_mean"] - df[f+"_mean_baseline"]
    return df

# ============================================================
# LABELS
# ============================================================
def add_labels(df):
    def group_map(u):
        if u in OUD_USERS:
            return "OUD"
        if u in CONTROL_USERS:
            return "Control"
        return None

    def res_map(u):
        if u in resilient_high:
            return "High"
        if u in resilient_low:
            return "Low"
        return None

    df["Group"] = df["user"].map(group_map)
    df["Resilience"] = df["user"].map(res_map)
    df = df.dropna(subset=["Group","Resilience"])

    df["Group"] = pd.Categorical(df["Group"], ["Control","OUD"])
    df["Resilience"] = pd.Categorical(df["Resilience"], ["High","Low"])
    return df

# ============================================================
# GEE & LMM (FEATURE SCREENING INCLUDED)
# ============================================================
def run_gee(task_df, feature_cols):
    rows = []
    for f in feature_cols:
        y = f + "_delta"
        sub = task_df[["user","Group","Resilience","n_windows",y]].dropna()

        if sub.shape[0] < 10:
            continue
        if sub[y].std() < MIN_STD:
            continue
        if sub["n_windows"].nunique() < 2:
            continue
        if np.abs(sub[y]).max() > Z_CLIP * 1.5:
            continue

        try:
            model = smf.gee(
                f"{y} ~ C(Resilience) + C(Group) + n_windows",
                groups="user",
                data=sub,
                family=sm.families.Gaussian(),
                cov_struct=sm.cov_struct.Exchangeable()
            )
            res = model.fit(cov_type="robust")
            key = "C(Resilience)[T.Low]"
            rows.append({
                "feature": f,
                "coef_low_vs_high": res.params[key],
                "se": res.bse[key],
                "p": res.pvalues[key],
                "ci_low": res.conf_int().loc[key][0],
                "ci_high": res.conf_int().loc[key][1]
            })
        except Exception:
            continue

    out = pd.DataFrame(rows)
    if len(out) > 0:
        out["q"] = multipletests(out["p"], method="fdr_bh")[1]
    return out.sort_values("q")

def run_lmm(task_df, feature_cols):
    rows = []
    for f in feature_cols:
        y = f + "_delta"
        sub = task_df[["user","Group","Resilience","n_windows",y]].dropna()
        if sub.shape[0] < 10 or sub[y].std() < MIN_STD:
            continue
        try:
            md = smf.mixedlm(
                f"{y} ~ C(Resilience) + C(Group) + n_windows",
                sub,
                groups=sub["user"]
            )
            res = md.fit(reml=False)
            key = "C(Resilience)[T.Low]"
            rows.append({
                "feature": f,
                "coef_low_vs_high": res.params.get(key, np.nan),
                "p": res.pvalues.get(key, np.nan)
            })
        except Exception:
            continue 
    return pd.DataFrame(rows)

# ============================================================
# MAIN
# ============================================================
def main():
    print("Loading windows...")
    win_df = load_windows()
    feature_cols = get_feature_cols(win_df)

    print(f"{len(feature_cols)} features detected.")

    print("Applying global z-score (RQ1)...")
    win_df = global_zscore(win_df, feature_cols)

    print("Aggregating to task-level...")
    task_df = aggregate_task_level(win_df, feature_cols)

    print("Computing jelly baseline deltas...")
    task_df = compute_baseline_delta(task_df, feature_cols)

    print("Adding labels...")
    task_df = add_labels(task_df)

    task_df.to_csv(f"{OUT_PREFIX}_task_level.csv", index=False)

    print("Running GEE (population-average)...")
    gee_res = run_gee(task_df, feature_cols)
    gee_res.to_csv(f"{OUT_PREFIX}_gee_results.csv", index=False)

    print("Running LMM (sensitivity)...")
    lmm_res = run_lmm(task_df, feature_cols)
    lmm_res.to_csv(f"{OUT_PREFIX}_lmm_results.csv", index=False)

    print("Done.")
    print("Outputs:")
    print(f" - {OUT_PREFIX}_task_level.csv")
    print(f" - {OUT_PREFIX}_gee_results.csv")
    print(f" - {OUT_PREFIX}_lmm_results.csv")

if __name__ == "__main__":
    main()
