import json
from simalign import SentenceAligner

# -------------------------
# Load English-Hindi Pairs
# -------------------------

def load_transcripts(filename):
    data = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if ':' in line:
                key, sentence = line.strip().split(":", 1)
                data[key.strip()] = sentence.strip()
    return data

english_data = load_transcripts("/content/transcripts.txt")
hindi_data = load_transcripts("/content/translated_hindi.txt")

# -------------------------
# Load ASR Word Alignments
# -------------------------

with open("/content/word_alignments.json", "r", encoding="utf-8") as f:
    asr_alignments = json.load(f)

# -------------------------
# Initialize SimAlign
# -------------------------

aligner = SentenceAligner(model="bert", token_type="bpe")

# -------------------------
# Map Hindi Words to English Words & Timestamps
# -------------------------

output = {}

for utt_id in english_data:
    if utt_id not in hindi_data or utt_id not in asr_alignments:
        continue

    eng_sent = english_data[utt_id]
    hin_sent = hindi_data[utt_id]
    word_alignments = asr_alignments[utt_id]["alignments"]

    eng_words = [item[0] for item in word_alignments]  # English words from JSON
    eng_word_timings = {i: item for i, item in enumerate(word_alignments)}

    hin_words = hin_sent.split()

    # Align using SimAlign
    alignment = aligner.get_word_aligns(eng_words, hin_words)
    aligned_pairs = alignment["inter"]

    mapped = []
    for eng_idx, hin_idx in aligned_pairs:
        eng_word, start, end, conf = eng_word_timings[eng_idx]
        hin_word = hin_words[hin_idx]
        mapped.append({
            "hindi_word": hin_word,
            "english_word": eng_word,
            "start": start,
            "end": end,
            "confidence": conf
        })

    output[utt_id] = mapped

# -------------------------
# Save or Print Output
# -------------------------

# Save to JSON
with open("aligned_output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print("✅ Alignment complete! Results saved to 'aligned_output.json'")
