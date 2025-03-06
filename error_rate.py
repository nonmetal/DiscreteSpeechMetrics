import os
import json
import argparse
import whisper
import jiwer
import librosa

# Function to compute WER (Word Error Rate) for English
def calculate_wer(ref_text, hyp_text):
    return jiwer.wer(ref_text.lower(), hyp_text.lower())

# Function to compute CER (Character Error Rate) for Japanese
def calculate_cer(ref_text, hyp_text):
    return jiwer.cer(ref_text, hyp_text)  # No lowercasing since Japanese is case-insensitive

def process_wav_files(wav_dir, json_path, is_japanese):
    # Load reference transcriptions
    with open(json_path, "r", encoding="utf-8") as f:
        reference_texts = json.load(f)

    # Load Whisper model
    model = whisper.load_model("base")

    # Store scores
    error_scores = []

    # Process each matched WAV file
    for file_name in sorted(os.listdir(wav_dir)):  # Sort for consistency
        if file_name.endswith(".wav"):
            file_path = os.path.join(wav_dir, file_name)
            file_key = file_name.replace(".wav", "")  # Extract key for JSON lookup

            if file_key in reference_texts:
                try:
                    # Load reference text
                    ref_text = reference_texts[file_key]["text"]

                    # Transcribe audio with Whisper
                    result = model.transcribe(file_path, language="ja" if is_japanese else None)
                    hyp_text = result["text"]

                    # Compute error rate
                    if is_japanese:
                        error_rate = calculate_cer(ref_text, hyp_text)
                        metric_name = "CER"
                    else:
                        error_rate = calculate_wer(ref_text, hyp_text)
                        metric_name = "WER"

                    error_scores.append(error_rate)

                    print(f"{file_name} | {metric_name}: {error_rate:.4f}")

                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

            else:
                print(f"Skipping {file_name}: No matching entry in JSON.")

    # Compute mean error rate
    if error_scores:
        mean_error = sum(error_scores) / len(error_scores)
        print(f"\nFinal {metric_name} Results:")
        print(f"Mean {metric_name}: {mean_error:.4f}")
    else:
        print("No valid scores computed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute WER (English) or CER (Japanese) using Whisper")
    parser.add_argument("wav_dir", type=str, help="Directory containing WAV files")
    parser.add_argument("json_path", type=str, help="Path to the JSON file with ground-truth transcripts")
    parser.add_argument("--ja", action="store_true", help="Enable this flag to compute Japanese Character Error Rate (CER) instead of WER")

    args = parser.parse_args()
    process_wav_files(args.wav_dir, args.json_path, args.ja)
