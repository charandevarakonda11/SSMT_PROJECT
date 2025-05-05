import whisperx
import torch

# 1. Load your model (use 'cuda' if you have GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisperx.load_model("large-v2", device)

# 2. Transcribe audio
audio_path = "your_audio_file.wav"  # Replace with your actual file path
transcription = model.transcribe(audio_path)

print("Initial transcription:")
print(transcription["text"])

# 3. Load alignment model and metadata
model_a, metadata = whisperx.load_align_model(language_code="en", device=device)

# 4. Align with word-level timestamps
aligned = whisperx.align(transcription["segments"], model_a, metadata, audio_path, device)

# 5. Print word-level alignments
print("\nWord-level alignments:")
for word_info in aligned["word_segments"]:
    print(f"{word_info['word']} - start: {word_info['start']:.2f}, end: {word_info['end']:.2f}")
