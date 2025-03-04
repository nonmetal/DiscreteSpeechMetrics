import os
import argparse
import numpy as np
import librosa
from discrete_speech_metrics import (
    SpeechBERTScore, SpeechBLEU, SpeechTokenDistance, MCD, LogF0RMSE, UTMOS
)
from scipy.io import wavfile

def process_wav_files(output_dir, reference_dir):
    # Initialize metric calculators
    bert_score = SpeechBERTScore(sr=16000, model_type="wavlm-large", layer=14, use_gpu=True)
    bleu_score = SpeechBLEU(sr=16000, model_type="hubert-base", vocab=200, layer=11, n_ngram=2, remove_repetition=True, use_gpu=True)
    token_distance = SpeechTokenDistance(sr=16000, model_type="hubert-base", vocab=200, layer=6, distance_type="jaro-winkler", remove_repetition=False, use_gpu=True)
    mcd_score = MCD(sr=16000)
    logf0rmse_score = LogF0RMSE(sr=16000)
    utmos_score = UTMOS(sr=16000)

    # Lists to store metric scores
    bert_scores, bleu_scores, token_distances = [], [], []
    mcd_scores, logf0rmse_scores, utmos_scores = [], [], []

    # Get all .wav files in output_dir
    gen_files = {f for f in os.listdir(output_dir) if f.endswith(".wav")}
    ref_files = {f for f in os.listdir(reference_dir) if f.endswith(".wav")}

    # Process only matching files
    matched_files = gen_files & ref_files

    for file_name in sorted(matched_files):  # Sort for consistency
        ref_file = os.path.join(reference_dir, file_name)
        gen_file = os.path.join(output_dir, file_name)

        try:
            # Load reference and generated audio (force mono)
            ref_audio, sr_ref = librosa.load(ref_file, sr=16000, mono=True)
            gen_audio, sr_gen = librosa.load(gen_file, sr=16000, mono=True)

            # Ensure neither signal is empty
            if len(ref_audio) == 0 or len(gen_audio) == 0:
                print(f"Skipping {file_name}: Empty audio detected.")
                continue

            # Pad the shorter signal with zeros
            max_length = max(len(ref_audio), len(gen_audio))
            ref_audio = np.pad(ref_audio, (0, max_length - len(ref_audio)), 'constant')
            gen_audio = np.pad(gen_audio, (0, max_length - len(gen_audio)), 'constant')

            # Compute SpeechBERTScore
            precision, _, _ = bert_score.score(ref_audio, gen_audio)
            bert_scores.append(precision)

            # Compute SpeechBLEU
            bleu = bleu_score.score(ref_audio, gen_audio)
            bleu_scores.append(bleu)

            # Compute SpeechTokenDistance
            distance = token_distance.score(ref_audio, gen_audio)
            token_distances.append(distance)

            # Compute MCD
            mcd = mcd_score.score(ref_audio, gen_audio)
            mcd_scores.append(mcd)

            # Compute Log F0 RMSE
            logf0rmse = logf0rmse_score.score(ref_audio, gen_audio)
            logf0rmse_scores.append(logf0rmse)

            # Compute UTMOS (does not require reference)
            utmos = utmos_score.score(gen_audio)
            utmos_scores.append(utmos)

            print(f"File: {file_name} | BERT: {precision:.4f} | BLEU: {bleu:.4f} | TokenDist: {distance:.4f} | MCD: {mcd:.4f} | LogF0RMSE: {logf0rmse:.4f} | UTMOS: {utmos:.4f}")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # Compute Mean and Standard Deviation
    def compute_mean_std(scores, name):
        if scores:
            mean_val = np.mean(scores)
            std_val = np.std(scores)
            print(f"{name} - Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        else:
            print(f"{name} - No valid scores computed.")

    print("\nFinal Results:")
    compute_mean_std(bert_scores, "SpeechBERTScore")
    compute_mean_std(bleu_scores, "SpeechBLEU")
    compute_mean_std(token_distances, "SpeechTokenDistance")
    compute_mean_std(mcd_scores, "MCD")
    compute_mean_std(logf0rmse_scores, "Log F0 RMSE")
    compute_mean_std(utmos_scores, "UTMOS")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Speech Metrics for Matched WAV Files")
    parser.add_argument("output_dir", type=str, help="Directory containing generated WAV files")
    parser.add_argument("reference_dir", type=str, help="Directory containing reference WAV files")

    args = parser.parse_args()
    process_wav_files(args.output_dir, args.reference_dir)