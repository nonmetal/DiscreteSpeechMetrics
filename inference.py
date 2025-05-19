import os
import argparse
import numpy as np
import librosa
from discrete_speech_metrics import (
    SpeechBERTScore, SpeechBLEU, SpeechTokenDistance, MCD, LogF0RMSE, SpeakerSimilarity, UTMOS, DNSMOS, AudioDuration  # [DNSMOS]
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
    dnsmos_score = DNSMOS(sr=16000)
    speaker_sim_score = SpeakerSimilarity(use_gpu=True)
    duration_scorer = AudioDuration(sr=16000)  # [DURATION]

    # Lists to store metric scores
    bert_scores, bleu_scores, token_distances = [], [], []
    mcd_scores, logf0rmse_scores, utmos_scores, dnsmos_scores, sim_scores, duration_errors = [], [], [], [], [], [], []

    # Get all .wav files in output_dir
    gen_files = {f for f in os.listdir(output_dir) if f.endswith(".wav")}
    ref_files = {f for f in os.listdir(reference_dir) if f.endswith(".wav")}
    matched_files = gen_files & ref_files

    for file_name in sorted(matched_files):
        ref_file = os.path.join(reference_dir, file_name)
        gen_file = os.path.join(output_dir, file_name)

        try:
            ref_audio, sr_ref = librosa.load(ref_file, sr=16000, mono=True)
            gen_audio, sr_gen = librosa.load(gen_file, sr=16000, mono=True)

            if len(ref_audio) == 0 or len(gen_audio) == 0:
                print(f"Skipping {file_name}: Empty audio detected.")
                continue

            max_length = max(len(ref_audio), len(gen_audio))
            ref_audio = np.pad(ref_audio, (0, max_length - len(ref_audio)), 'constant')
            gen_audio = np.pad(gen_audio, (0, max_length - len(gen_audio)), 'constant')

            # Compute Metrics
            precision, _, _ = bert_score.score(ref_audio, gen_audio)
            bleu = bleu_score.score(ref_audio, gen_audio)
            distance = token_distance.score(ref_audio, gen_audio)
            mcd = mcd_score.score(ref_audio, gen_audio)
            logf0rmse = logf0rmse_score.score(ref_audio, gen_audio)
            utmos = utmos_score.score(gen_audio)
            dns = dnsmos_score.score(gen_audio)
            sim = speaker_sim_score.compute_similarity(ref_audio, gen_audio)
            duration_error_pct = duration_scorer.score(ref_audio, gen_audio)  # [DURATION]

            # Collect Scores
            bert_scores.append(precision)
            bleu_scores.append(bleu)
            token_distances.append(distance)
            mcd_scores.append(mcd)
            logf0rmse_scores.append(logf0rmse)
            utmos_scores.append(utmos)
            dnsmos_scores.append(dns)
            sim_scores.append(sim)
            duration_errors.append(duration_error_pct)  # [DURATION]

            print(f"File: {file_name} | BERT: {precision:.4f} | BLEU: {bleu:.4f} | TokenDist: {distance:.4f} | "
                  f"MCD: {mcd:.4f} | LogF0RMSE: {logf0rmse:.4f} | UTMOS: {utmos:.4f} | DNSMOS: {dns:.4f} | "
                  f"SIM: {sim:.4f} | Duration Error: {duration_error_pct:.2f}%")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # Summary Function
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
    compute_mean_std(dnsmos_scores, "DNSMOS")
    compute_mean_std(sim_scores, "SpeakerSimilarity")
    compute_mean_std(duration_errors, "Audio Duration Error (%)")  # [DURATION]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Speech Metrics for Matched WAV Files")
    parser.add_argument("output_dir", type=str, help="Directory containing generated WAV files")
    parser.add_argument("reference_dir", type=str, help="Directory containing reference WAV files")

    args = parser.parse_args()
    process_wav_files(args.output_dir, args.reference_dir)
