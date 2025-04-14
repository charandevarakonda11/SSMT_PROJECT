import os
import json
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from python_speech_features import mfcc, delta

def is_frame_stressed(start_time, end_time, stress_regions):
    for region_start, region_end in stress_regions:
        if start_time < region_end and end_time > region_start:
            return 1
    return 0

def compute_sdc(mfcc_feat, d=1, p=3, k=7):
    N, D = mfcc_feat.shape
    sdc_feat = []
    deltas = delta(mfcc_feat, N=2)
    for n in range(N):
        vec = []
        for i in range(k):
            idx = n + i * p
            if idx < len(deltas):
                vec.extend(deltas[idx])
            else:
                vec.extend([0.0] * D)
        sdc_feat.append(vec)
    return np.array(sdc_feat)

def extract_features(y, sr, hop_length, win_length):
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=75, fmax=500, sr=sr, frame_length=win_length, hop_length=hop_length)
    energy = librosa.feature.rms(y=y, frame_length=win_length, hop_length=hop_length)[0]
    mfcc_feat = mfcc(y, sr, winlen=win_length/sr, winstep=hop_length/sr, nfft=2048)
    sdc_feat = compute_sdc(mfcc_feat)
    return f0, energy, mfcc_feat, sdc_feat

def process_audio_file(audio_path, audio_name, stress_regions, total_duration, all_f0_energy, all_full_features):
    y, sr = librosa.load(audio_path, sr=8000)
    hop_length = 512
    win_length = 1024

    f0, energy, mfcc_feat, sdc_feat = extract_features(y, sr, hop_length, win_length)
    frame_duration = hop_length / sr
    frame_count = min(len(energy), len(mfcc_feat), len(sdc_feat), len(f0))

    for i in range(frame_count):
        start_time = i * frame_duration
        end_time = start_time + frame_duration
        label = is_frame_stressed(start_time, end_time, stress_regions)

        f0_val = None if f0[i] is None else float(f0[i])
        energy_val = float(energy[i])

        frame_common = {
            "audio_name": audio_name,
            "start": start_time,
            "end": end_time,
            "f0": f0_val,
            "energy": energy_val,
            "label": label
        }

        # Add to simple f0+energy set
        all_f0_energy.append(frame_common)

        # Add to detailed set
        frame_detailed = {
            **frame_common,
            "mfcc": mfcc_feat[i].tolist(),
            "sdc": sdc_feat[i].tolist()
        }
        all_full_features.append(frame_detailed)

def main(audio_folder, csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    all_f0_energy = []
    all_full_features = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio_name = row['audio_name']
        audio_path = os.path.join(audio_folder, audio_name + ".wav")
        try:
            stress_regions = eval(row['stress_regions']) if pd.notna(row['stress_regions']) else []
            duration = float(row['length'])

            process_audio_file(audio_path, audio_name, stress_regions, duration, all_f0_energy, all_full_features)
        except Exception as e:
            print(f"Error processing {audio_name}: {e}")

    # Save final 2 JSON files
    with open(os.path.join(output_dir, "f0_energy.json"), 'w') as f:
        json.dump(all_f0_energy, f, indent=2)

    with open(os.path.join(output_dir, "f0_energy_mfcc_sdc.json"), 'w') as f:
        json.dump(all_full_features, f, indent=2)

if __name__ == "__main__":
    audio_folder = "/scratch/data_temp/ssmt_pr/FINAL_DATA_ENGLISH"        # Update this path
    csv_path = "/home2/sricharan.d/SSMT/pyfiles/Stress.csv"              # Update this path
    output_dir = "/scratch/data_temp/ssmt_pr/Json_files"                 # Update this path
    main(audio_folder, csv_path, output_dir)
