import os
import glob
import numpy as np
import pandas as pd
import neurokit2 as nk
import flirt
from collections import Counter
import traceback

# ===============================
# Config (Empatica E4)
# ===============================
SR_EDA  = 4
SR_PPG  = 64
SR_ACC  = 32
SR_HR   = 1
SR_TEMP = 4

WINDOW = 30   # seconds
STEP   = 15   # seconds

SAVE_DIR = "./save_right"
os.makedirs(SAVE_DIR, exist_ok=True)

CALM_TASKS   = {"jelly", "count", "baseline", "good", "animal"}
STRESS_TASKS = {"prepareSong", "arithmetic", "bad", "stroop", "stress"}

# HRV quality
MIN_RR_COUNT = 8


# ===============================
# Label helpers
# ===============================
def majority_vote_task(labels):
    labels = [l for l in labels if pd.notna(l) and l != "None"]
    if len(labels) == 0:
        return "Unknown"
    return Counter(labels).most_common(1)[0][0]


def task_to_state(task):
    if task in CALM_TASKS:
        return 0
    if task in STRESS_TASKS:
        return 1
    return -1


def get_craving_dict(excel_path="./craving_label.xlsx"):
    df = pd.read_excel(excel_path, index_col=0)
    cell_dict = {}
    for row_key, row in df.iterrows():
        for col_name, value in row.items():
            if pd.notna(value) and value != "":
                cell_dict[(str(row_key), str(col_name))] = value
    return cell_dict


# ===============================
# Preprocess
# ===============================
def preprocess_eda_ppg(eda_raw, bvp_raw):
    eda_rs = nk.signal_resample(
        eda_raw, sampling_rate=SR_EDA, desired_sampling_rate=SR_PPG, method="poly"
    )
    eda_clean = nk.eda_clean(eda_rs, sampling_rate=SR_PPG)
    eda_df, _ = nk.eda_process(eda_clean, sampling_rate=SR_PPG)

    bvp_df, _ = nk.ppg_process(bvp_raw, sampling_rate=SR_PPG)

    hr_1hz = nk.signal_resample(
        bvp_df["PPG_Rate"],
        sampling_rate=SR_PPG,
        desired_sampling_rate=SR_HR,
        method="mean"
    )

    return eda_df, bvp_df, hr_1hz


# ===============================
# FLIRT (single window)
# ===============================
def flirt_block_single_window(data, cols, prefix, data_frequency):
    df = pd.DataFrame(data, columns=cols).fillna(0)
    if len(df) < 2:
        raise RuntimeError("Too few samples for FLIRT")

    feats = flirt.get_acc_features(
        df,
        window_length=len(df),
        window_step_size=len(df),
        data_frequency=data_frequency,
    )

    if len(feats) != 1:
        raise RuntimeError(f"FLIRT returned {len(feats)} windows")

    return feats.add_prefix(prefix).reset_index(drop=True)


# ===============================
# HRV
# ===============================
def compute_hrv_from_peaks(peaks_bool):
    peaks_bool = np.asarray(peaks_bool)
    if peaks_bool.size == 0:
        raise RuntimeError("Empty PPG_Peaks")

    peak_idx = np.where(peaks_bool)[0]
    if len(peak_idx) < MIN_RR_COUNT + 1:
        raise RuntimeError(f"Too few peaks: {len(peak_idx)}")

    # ---- peak fixing ----
    try:
        _, peak_idx = nk.signal_fixpeaks(
            peak_idx,
            sampling_rate=SR_PPG,
            method="Kubios"
        )
    except Exception as e:
        raise RuntimeError(f"signal_fixpeaks failed") from e

    peak_idx = np.asarray(peak_idx)
    rr_count = len(peak_idx) - 1
    if rr_count < MIN_RR_COUNT:
        raise RuntimeError(f"Too few RR intervals: {rr_count}")

    if np.any(np.diff(peak_idx) <= 0):
        raise RuntimeError("Non-monotonic peak indices")

    # ---- HRV time ----
    try:
        hrv_time = nk.hrv_time(
            peak_idx,
            sampling_rate=SR_PPG,
            show=False
        )
    except Exception as e:
        raise RuntimeError("hrv_time failed") from e

    # ---- HRV freq ----
    try:
        hrv_freq = nk.hrv_frequency(
            peak_idx,
            sampling_rate=SR_PPG,
            psd_method="lomb",
            show=False
        )
    except Exception as e:
        raise RuntimeError("hrv_frequency failed") from e

    if hrv_time.empty or hrv_freq.empty:
        raise RuntimeError("Empty HRV output")

    return hrv_time, hrv_freq, rr_count


# ===============================
# Feature extraction
# ===============================
def extract_features(user, eda_df, bvp_df, hr_1hz,
                     temp_df, acc_df, label_series, craving_dict):

    rows = []

    acc_x = acc_df["acc0"].to_numpy()
    acc_y = acc_df["acc1"].to_numpy()
    acc_z = acc_df["acc2"].to_numpy()
    temp  = temp_df["temp"].to_numpy()

    total_seconds = len(hr_1hz)
    theoretical_windows = max(
        0, int((total_seconds - WINDOW) / STEP) + 1
    )

    for i in range(theoretical_windows):
        start = i * STEP
        end   = start + WINDOW

        eda_win = eda_df.iloc[start*SR_PPG : end*SR_PPG]
        ppg_win = bvp_df.iloc[start*SR_PPG : end*SR_PPG]
        hr_win  = hr_1hz[start : end]

        temp_win = temp[start*SR_TEMP : end*SR_TEMP]
        x = acc_x[start*SR_ACC : end*SR_ACC]
        y = acc_y[start*SR_ACC : end*SR_ACC]
        z = acc_z[start*SR_ACC : end*SR_ACC]

        if min(len(eda_win), len(ppg_win), len(hr_win)) == 0:
            continue

        labels_win = label_series.iloc[start*SR_EDA : end*SR_EDA]
        task = majority_vote_task(labels_win)
        state = task_to_state(task)

        craving_raw = craving_dict.get((str(user), str(task)), 0)
        craving = int(craving_raw > 0)

        tonic  = eda_win["EDA_Tonic"]
        phasic = eda_win["EDA_Phasic"]

        num_onset = eda_win["SCR_Onsets"].sum()
        num_peaks = eda_win["SCR_Peaks"].sum()
        num_recovery = eda_win["SCR_Recovery"].sum()

        EDA_height = eda_win["SCR_Height"].sum()
        EDA_amplitude = eda_win["SCR_Amplitude"].sum()
        EDA_risetime = eda_win["SCR_RiseTime"].sum()
        EDA_recoverytime = eda_win["SCR_RecoveryTime"].sum()

  
        try:
            hrv_time, hrv_freq, rr_count = compute_hrv_from_peaks(
                ppg_win["PPG_Peaks"].values
            )
        except Exception:
            continue

        try:
            acc_f = flirt_block_single_window(np.c_[x,y,z], ["x","y","z"], "ACC_", SR_ACC)
            temp_f = flirt_block_single_window(temp_win.reshape(-1,1), ["TEMP"], "TEMP_", SR_TEMP)
            hr_f   = flirt_block_single_window(hr_win.reshape(-1,1), ["HR"], "HR_", SR_HR)
            bvp_f  = flirt_block_single_window(
                ppg_win["PPG_Clean"].values.reshape(-1,1),
                ["BVP"], "BVP_", SR_PPG
            )
            tonic_f  = flirt_block_single_window(tonic.values.reshape(-1,1), ["tonic"], "Tonic_", SR_PPG)
            phasic_f = flirt_block_single_window(phasic.values.reshape(-1,1), ["phasic"], "Phasic_", SR_PPG)
        except Exception:
            continue

        result = pd.concat(
            [
                bvp_f, acc_f, hr_f, temp_f,
                tonic_f, phasic_f,
                hrv_freq.reset_index(drop=True),
                hrv_time.reset_index(drop=True)
            ],
            axis=1
        )

        result["EDA_onset"] = num_onset
        result["EDA_peaks"] = num_peaks
        result["EDA_recovery"] = num_recovery
        result["EDA_height"] = EDA_height
        result["EDA_amplitude"] = EDA_amplitude
        result["EDA_risetime"] = EDA_risetime
        result["EDA_recoverytime"] = EDA_recoverytime

        result.insert(0, "rr_count", rr_count)
        result.insert(0, "stress", state)
        result.insert(0, "task", task)
        result.insert(0, "craving", craving)
        result.insert(0, "window_start_sec", start)
        result.insert(0, "user", user)

        rows.append(result)

    return rows, theoretical_windows


# ===============================
# Main
# ===============================
def main():
    craving_dict = get_craving_dict("./craving_label.xlsx")

    acc_files = []
    acc_files += glob.glob("./raw_data/student/left*acc.csv")
    acc_files += glob.glob("./raw_data/OUD/left*acc.csv")

    for f in acc_files:
        try:
            user = f.split("/")[-1].split("_")[1]
            if user in EXCLUDE_USERS:
                continue
    
            print(f"\nProcessing {user}")
    
            eda_raw = pd.read_csv(f.replace("acc", "eda"))["eda"].values
            bvp_raw = pd.read_csv(f.replace("acc", "bvp"))["bvp"].values
            labels  = pd.read_csv(f.replace("acc", "eda"))["label"]
    
            temp_df = pd.read_csv(f.replace("acc", "temp"))
            acc_df  = pd.read_csv(f)
    
            eda_df, bvp_df, hr_1hz = preprocess_eda_ppg(eda_raw, bvp_raw)
    
            features, _ = extract_features(
                user, eda_df, bvp_df, hr_1hz, temp_df, acc_df, labels, craving_dict
            )
    
            if len(features) == 0:
                print("No valid windows")
                continue
    
            out_df = pd.concat(features, ignore_index=True)
            out_df.to_csv(os.path.join(SAVE_DIR, f"data_{user}.csv"), index=False)
    
            print("Task distribution:")
            print(out_df["task"].value_counts())

        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    main()
