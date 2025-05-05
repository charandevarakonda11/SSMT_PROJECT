import whisperx
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


model = whisperx.load_model("medium", device, language="hi", compute_type="float32")


audio_path = "/content/1746305558.wav"
transcription = model.transcribe(audio_path)

print("Transcription:")
print(transcription)

# 3. Load alignment model and metadata
align_model, metadata = whisperx.load_align_model(language_code="hi", device=device)

# 4. Perform word alignment
aligned_result = whisperx.align(transcription["segments"], align_model, metadata, audio_path, device)

# 5. Print word-level timestamps
for word in aligned_result["word_segments"]:
    print(f"{word['word']} [{word['start']:.2f}s - {word['end']:.2f}s]")
