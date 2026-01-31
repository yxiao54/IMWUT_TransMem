import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr

from sklearn.preprocessing import StandardScaler

from sklearn.cross_decomposition import PLSRegression

# =========================
# Config
# =========================
EMB_NPZ = "emb_data.npz"     # contains X_gb, X_bl
LABELS_CSV = "labels.csv"   # contains hr_auc

N_PERM = 5000
RANDOM_SEED = 0

np.random.seed(RANDOM_SEED)

# =========================
# Load data
# =========================
emb = np.load(EMB_NPZ)
X_gb = emb["X_gb"]   # (N, D1)
X_bl = emb["X_bl"]   # (N, D2)

labels = pd.read_csv(LABELS_CSV)
y_hr = labels["hr_auc"].values

assert X_gb.shape[0] == y_hr.shape[0]
assert X_bl.shape[0] == y_hr.shape[0]

N = y_hr.shape[0]
print(f"Loaded N = {N}")

# =========================
# Utility: permutation test
# =========================
def spearman_permutation(x, y, n_perm=N_PERM, seed=RANDOM_SEED, desc=""):
    r_obs, p_obs = spearmanr(x, y)

    rng = np.random.default_rng(seed)
    perm_rs = np.zeros(n_perm)

    for i in tqdm(range(n_perm), desc=desc, leave=False):
        y_perm = rng.permutation(y)
        perm_rs[i], _ = spearmanr(x, y_perm)

    p_perm = np.mean(np.abs(perm_rs) >= np.abs(r_obs))
    return r_obs, p_obs, p_perm



def run_pls(X, y, name):
    Xs = StandardScaler().fit_transform(X)

    pls = PLSRegression(n_components=1)
    z = pls.fit_transform(Xs, y)[0][:, 0]

    r, p, p_perm = spearman_permutation(
        z, y, desc=f"PLS permuting {name}"
    )

    return {
        "representation": name,
        "method": "PLS",
        "spearman_r": r,
        "spearman_p": p,
        "perm_p": p_perm,
        
    }

# =========================
# Run  analyses
# =========================
results = []

for X, name in [(X_gb, "Good+Bad"), (X_bl, "Baseline")]:

    results.append(run_pls(X, y_hr, name))


# =========================
# Save & report
# =========================
df = pd.DataFrame(results)
df.to_csv("rq2_hr_latent_analysis.csv", index=False)

print("\n=== HR latent representation analysis ===")
print(df)
print("\nSaved to rq2_hr_latent_analysis.csv")
