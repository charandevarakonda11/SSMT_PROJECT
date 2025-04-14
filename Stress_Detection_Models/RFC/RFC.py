import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
# import xgboost as xgb


def get_windowed_features(i, feat_array):
    feat_dim = feat_array.shape[1]
    prev_feat = feat_array[i - 1] if i > 0 else np.zeros(feat_dim)
    curr_feat = feat_array[i]
    next_feat = feat_array[i + 1] if i < len(feat_array) - 1 else np.zeros(feat_dim)
    return np.concatenate([prev_feat, curr_feat, next_feat])


def load_and_prepare_features(f0_energy_json, full_feature_json):
    # Load JSON data
    with open(f0_energy_json, 'r') as f:
        f0_energy_raw = json.load(f)
    with open(full_feature_json, 'r') as f:
        full_feature_raw = json.load(f)

    # Convert to DataFrames for easier processing
    df_f0_energy = pd.DataFrame(f0_energy_raw)
    df_full = pd.DataFrame(full_feature_raw)

    # Sort by audio and frame start time (optional but useful)
    df_f0_energy.sort_values(by=["audio_name", "start"], inplace=True)
    df_full.sort_values(by=["audio_name", "start"], inplace=True)

    f0_energy_data = []
    full_feature_data = []

    for audio_name, group in df_f0_energy.groupby("audio_name"):
        fe = group[["f0", "energy"]].fillna(0.0).to_numpy(dtype=np.float32)
        labels = group["label"].to_numpy(dtype=int)
        for i in range(len(group)):
            fe_window = get_windowed_features(i, fe)
            f0_energy_data.append(np.append(fe_window, labels[i]))

    for audio_name, group in df_full.groupby("audio_name"):
        fe = group[["f0", "energy"]].fillna(0.0).to_numpy(dtype=np.float32)
        mfcc = np.array(group["mfcc"].tolist(), dtype=np.float32)
        sdc = np.array(group["sdc"].tolist(), dtype=np.float32)
        labels = group["label"].to_numpy(dtype=int)

        for i in range(len(group)):
            fe_window = get_windowed_features(i, fe)
            mfcc_window = get_windowed_features(i, mfcc)
            sdc_window = get_windowed_features(i, sdc)
            combined = np.concatenate([fe_window, mfcc_window, sdc_window])
            full_feature_data.append(np.append(combined, labels[i]))

    return np.array(f0_energy_data), np.array(full_feature_data)


def main(json_dir):
    f0_energy_json = os.path.join(json_dir, "f0_energy.json")
    full_feature_json = os.path.join(json_dir, "f0_energy_mfcc_sdc.json")

    print("Loading features from JSON...")
    f0_energy_data, full_feature_data = load_and_prepare_features(f0_energy_json, full_feature_json)

    # Train model on f0 + energy
    print("\nTraining model with windowed f0 + energy features...")
    X1 = f0_energy_data[:, :-1]
    y1 = f0_energy_data[:, -1]
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

    # clf1 = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0, random_state=42)
    clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf1.fit(X1_train, y1_train)
    y1_pred = clf1.predict(X1_test)
    print("Windowed f0 + energy Model Accuracy:", accuracy_score(y1_test, y1_pred))
    print(classification_report(y1_test, y1_pred))

    # Train model on full feature set
    print("\nTraining model with windowed full features (f0 + energy + MFCC + SDC)...")
    X2 = full_feature_data[:, :-1]
    y2 = full_feature_data[:, -1]
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

    clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf2.fit(X2_train, y2_train)
    y2_pred = clf2.predict(X2_test)
    print("Windowed Full Feature Model Accuracy:", accuracy_score(y2_test, y2_pred))
    print(classification_report(y2_test, y2_pred))


if __name__ == "__main__":
    json_dir = "/scratch/data_temp/ssmt_pr/Json_files"  # Update if needed
    main(json_dir)
