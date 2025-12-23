import json

#Temppppp

# Load input JSON files
with open("utterances.json", "r") as f:
    utterances = json.load(f)

with open("alignments.json", "r") as f:
    alignments = json.load(f)

# Function to compute average of a key
def avg(lst, idx):
    return sum(x[idx] for x in lst) / len(lst) if lst else 0

# Final result
scaling_factors = {}

for utt_id, utt_info in utterances.items():
    stressed_words = set(w.strip(".,!?").lower() for w in utt_info["stressed_words"])

    if utt_id not in alignments:
        continue

    alignment = alignments[utt_id]["alignments"]

    stressed_feats = []
    unstressed_feats = []

    for word, start, end, pitch in alignment:
        clean_word = word.strip(".,!?").lower()
        duration = end - start
        energy = pitch * 0.3  # Example heuristic (replace with real energy if available)

        features = (pitch, energy, duration)

        if clean_word in stressed_words:
            stressed_feats.append(features)
        else:
            unstressed_feats.append(features)

    # Calculate scaling factors if valid data is available
    if stressed_feats and unstressed_feats:
        pitch_sf = avg(stressed_feats, 0) / avg(unstressed_feats, 0)
        energy_sf = avg(stressed_feats, 1) / avg(unstressed_feats, 1)
        duration_sf = avg(stressed_feats, 2) / avg(unstressed_feats, 2)
    else:
        pitch_sf = energy_sf = duration_sf = 1.0  # Default if missing data

    scaling_factors[utt_id] = {
        "pitch_scaling_factor": round(pitch_sf, 2),
        "energy_scaling_factor": round(energy_sf, 2),
        "duration_scaling_factor": round(duration_sf, 2),
    }

# Save to JSON
with open("scaling_factors.json", "w") as f:
    json.dump(scaling_factors, f, indent=4)

print("Saved scaling factors to scaling_factors.json")
