import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import shuffle


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
    full_feature_data = []

    for _, group in df_f0_energy.groupby("audio_name"):
        fe = group[["f0", "energy"]].fillna(0.0).to_numpy(dtype=np.float32)
        labels = group["label"].to_numpy(dtype=int)
        for i in range(len(group)):
            fe_window = get_windowed_features(i, fe)
            f0_energy_data.append(np.append(fe_window, labels[i]))

    for _, group in df_full.groupby("audio_name"):
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


def run_label_propagation(X, y, y_true=None, description=""):
    print(f"\nRunning Label Propagation on {description}...")

    # Split into train/test
    X_train, X_test, y_train_true, y_test_true = train_test_split(
        X, y_true if y_true is not None else y, test_size=0.2, random_state=42
    )

    # Mask 70% of train labels to simulate unlabeled data
    rng = np.random.RandomState(42)
    n_total = len(y_train_true)
    n_labeled = int(n_total * 0.3)

    indices = np.arange(n_total)
    rng.shuffle(indices)

    y_train = np.full_like(y_train_true, fill_value=-1)
    y_train[indices[:n_labeled]] = y_train_true[indices[:n_labeled]]

    # Train Label Propagation
    model = LabelPropagation(kernel='rbf', gamma=20)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy on test set:", accuracy_score(y_test_true, y_pred))
    print(classification_report(y_test_true, y_pred))


def main(json_dir):
    f0_energy_json = os.path.join(json_dir, "f0_energy.json")
    full_feature_json = os.path.join(json_dir, "f0_energy_mfcc_sdc.json")

    print("Loading features from JSON...")
    f0_energy_data, full_feature_data = load_and_prepare_features(f0_energy_json, full_feature_json)

    subset_size = 20000

    # Prepare data (f0 + energy)
    X1 = f0_energy_data[:, :-1]
    y1 = f0_energy_data[:, -1].astype(int)
    X1, y1 = shuffle(X1, y1, random_state=42)

    X1_small = X1[:subset_size]
    y1_small = y1[:subset_size]
    y1_masked = np.copy(y1_small)
    y1_masked[1000:] = -1

    run_label_propagation(X1_small, y1_masked, y_true=y1_small, description="windowed f0 + energy")

    # Prepare data (f0 + energy + MFCC + SDC)
    X2 = full_feature_data[:, :-1]
    y2 = full_feature_data[:, -1].astype(int)
    X2, y2 = shuffle(X2, y2, random_state=42)

    X2_small = X2[:subset_size]
    y2_small = y2[:subset_size]
    y2_masked = np.copy(y2_small)
    y2_masked[1000:] = -1

    run_label_propagation(X2_small, y2_masked, y_true=y2_small, description="windowed full features (f0 + energy + MFCC + SDC)")


if __name__ == "__main__":
    json_dir = "/scratch/data_temp/ssmt_pr/Json_files"  # Update this path as needed
    main(json_dir)
