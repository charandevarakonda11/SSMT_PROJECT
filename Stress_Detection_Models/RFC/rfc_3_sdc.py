import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE


def get_windowed_features(i, feat_array):
    feat_dim = feat_array.shape[1]
    prev_feat = feat_array[i - 1] if i > 0 else np.zeros(feat_dim)
    curr_feat = feat_array[i]
    next_feat = feat_array[i + 1] if i < len(feat_array) - 1 else np.zeros(feat_dim)
    return np.concatenate([prev_feat, curr_feat, next_feat])


def load_and_prepare_features(f0_energy_json, full_feature_json):
    with open(f0_energy_json, 'r') as f:
        f0_energy_raw = json.load(f)
    with open(full_feature_json, 'r') as f:
        full_feature_raw = json.load(f)

    df_f0_energy = pd.DataFrame(f0_energy_raw)
    df_full = pd.DataFrame(full_feature_raw)

    df_f0_energy.sort_values(by=["audio_name", "start"], inplace=True)
    df_full.sort_values(by=["audio_name", "start"], inplace=True)

    f0_energy_data = []
    mfcc_data = []
    sdc_data = []
    full_feature_data = []

    for audio_name, group in df_full.groupby("audio_name"):
        fe = group[["f0", "energy"]].fillna(0.0).to_numpy(dtype=np.float32)
        mfcc = np.array(group["mfcc"].tolist(), dtype=np.float32)
        sdc = np.array(group["sdc"].tolist(), dtype=np.float32)
        labels = group["label"].to_numpy(dtype=int)

        for i in range(len(group)):
            fe_window = get_windowed_features(i, fe)
            mfcc_window = get_windowed_features(i, mfcc)
            sdc_window = get_windowed_features(i, sdc)

            # Individual
            f0_energy_data.append(np.append(fe_window, labels[i]))
            mfcc_data.append(np.append(mfcc_window, labels[i]))
            sdc_data.append(np.append(sdc_window, labels[i]))

            # Combined
            combined = np.concatenate([fe_window, mfcc_window, sdc_window])
            full_feature_data.append(np.append(combined, labels[i]))

    return (np.array(f0_energy_data),
            np.array(mfcc_data),
            np.array(sdc_data),
            np.array(full_feature_data))


def apply_smote(X, y):
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"Shape after SMOTE: {X_res.shape}")
    return X_res, y_res


def train_and_evaluate_model(X, y, label):
    print(f"\nTraining model with {label} features...")
    X_bal, y_bal = apply_smote(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"{label} Model Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def main(json_dir):
    f0_energy_json = os.path.join(json_dir, "f0_energy.json")
    full_feature_json = os.path.join(json_dir, "f0_energy_mfcc_sdc.json")

    print("Loading features from JSON...")
    f0_energy_data, mfcc_data, sdc_data, full_feature_data = load_and_prepare_features(
        f0_energy_json, full_feature_json
    )

    # Train individual models
    train_and_evaluate_model(f0_energy_data[:, :-1], f0_energy_data[:, -1], "Windowed f0 + energy")
    train_and_evaluate_model(mfcc_data[:, :-1], mfcc_data[:, -1], "Windowed MFCC")
    train_and_evaluate_model(sdc_data[:, :-1], sdc_data[:, -1], "Windowed SDC")
    train_and_evaluate_model(full_feature_data[:, :-1], full_feature_data[:, -1], "Windowed Full Feature")


if __name__ == "__main__":
    json_dir = "/scratch/data_temp/ssmt_pr/Json_files"  # Update this if needed
    main(json_dir)
