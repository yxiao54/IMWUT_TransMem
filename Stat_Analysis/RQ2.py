import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from scipy.stats import spearmanr
import statsmodels.formula.api as smf
from tqdm import tqdm

# =========================
# Config
# =========================
EMB_NPZ = "emb_data.npz"     # contains X_gb (N,3072), X_bl (N,1536)
LABELS_CSV = "labels.csv"   # contains user, eda_auc, hr_auc
N_PCA = 3                   # number of PCs to use in regression
N_PERM = 2000
N_BOOT = 2000
RANDOM_SEED = 0

# =========================
# Load data
# =========================
np.random.seed(RANDOM_SEED)

data = np.load(EMB_NPZ)
X_gb = data["X_gb"]
X_bl = data["X_bl"]

labels = pd.read_csv(LABELS_CSV)

assert X_gb.shape[0] == labels.shape[0]
assert X_bl.shape[0] == labels.shape[0]

N = labels.shape[0]
print(f"Loaded N = {N}")

# =========================
# Helper: run one regression
# =========================
def run_regression(X, y, label_name, outcome_name):
    """
    X: (N, D) embedding matrix
    y: (N,) outcome array
    """
    # ---- PCA ----
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=N_PCA, random_state=RANDOM_SEED)
    X_pcs = pca.fit_transform(Xs)

    df = pd.DataFrame(X_pcs, columns=[f"PC{i+1}" for i in range(N_PCA)])
    df[outcome_name] = y

    # ---- OLS ----
    formula = outcome_name + " ~ " + " + ".join(df.columns[:-1])
    model = smf.ols(formula=formula, data=df).fit(cov_type="HC3")

    beta_pc1 = model.params["PC1"]

    # ---- permutation test (PC1 coefficient) ----
    rng = np.random.default_rng(RANDOM_SEED)
    perm_betas = []

    for _ in tqdm(range(N_PERM), desc=f"Permuting {label_name} ? {outcome_name}", leave=False):
        y_perm = rng.permutation(y)
        df_perm = df.copy()
        df_perm[outcome_name] = y_perm
        m_perm = smf.ols(formula=formula, data=df_perm).fit()
        perm_betas.append(m_perm.params["PC1"])

    perm_betas = np.array(perm_betas)
    perm_p = np.mean(np.abs(perm_betas) >= np.abs(beta_pc1))

    # ---- bootstrap CI ----
    boot_betas = []
    for _ in tqdm(range(N_BOOT), desc=f"Bootstrap {label_name} ? {outcome_name}", leave=False):
        idx = resample(np.arange(N), replace=True, n_samples=N)
        df_boot = df.iloc[idx].reset_index(drop=True)
        try:
            m_boot = smf.ols(formula=formula, data=df_boot).fit()
            boot_betas.append(m_boot.params["PC1"])
        except Exception:
            continue

    boot_betas = np.array(boot_betas)
    ci_low, ci_high = np.percentile(boot_betas, [2.5, 97.5])

    # ---- Spearman (for reference) ----
    rho, rho_p = spearmanr(df["PC1"], y)

    return {
        "representation": label_name,
        "outcome": outcome_name,
        "beta_PC1": beta_pc1,
        "perm_p": perm_p,
        "boot_ci_low": ci_low,
        "boot_ci_high": ci_high,
        "spearman_r": rho,
        "spearman_p": rho_p,
        "R2": model.rsquared
    }

# =========================
# Run 4 regressions
# =========================
results = []

# 1) Good+Bad ? EDA
results.append(
    run_regression(
        X_gb,
        labels["eda_auc"].values,
        label_name="Good+Bad",
        outcome_name="eda_auc"
    )
)

# 2) Good+Bad ? HR
results.append(
    run_regression(
        X_gb,
        labels["hr_auc"].values,
        label_name="Good+Bad",
        outcome_name="hr_auc"
    )
)

# 3) Baseline ? EDA
results.append(
    run_regression(
        X_bl,
        labels["eda_auc"].values,
        label_name="Baseline",
        outcome_name="eda_auc"
    )
)

# 4) Baseline ? HR
results.append(
    run_regression(
        X_bl,
        labels["hr_auc"].values,
        label_name="Baseline",
        outcome_name="hr_auc"
    )
)

# =========================
# Save results
# =========================
results_df = pd.DataFrame(results)
results_df.to_csv("rq2_four_regressions_results.csv", index=False)

print("\n=== RQ2 Four Regression Results ===")
print(results_df)
print("\nSaved to rq2_four_regressions_results.csv")
