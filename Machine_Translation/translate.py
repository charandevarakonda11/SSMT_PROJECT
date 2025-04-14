from transformers import MarianMTModel, MarianTokenizer
import re

# Load the model and tokenizer for English to Hindi translation
model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_to_hindi(text):
    """Translates English text to Hindi using Helsinki-NLP/opus-mt-en-hi."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    hindi_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return hindi_text[0]

def process_file(input_file, output_file):
    """Reads a text file, translates each line while preserving file names, and writes to output."""
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            match = re.match(r"^(iiith_ied_stress_\d+):\s*(.+)", line.strip())  # Extract file name
            if match:
                file_name, english_text = match.groups()
                hindi_translation = translate_to_hindi(english_text)
                outfile.write(f"{file_name}: {hindi_translation}\n")  # Preserve file name
            else:
                outfile.write(line)  # If format doesn't match, write as is

# Example Usage
input_file = "/home/sricharan/Documents/SSMT/pyfiles/MT/transcripts.txt"  # Change to your actual input file
output_file = "translated_hindi.txt"  # Change to your desired output file
process_file(input_file, output_file)

print("Translation completed. Check output.txt")
