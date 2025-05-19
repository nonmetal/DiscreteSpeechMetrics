import torch
import numpy as np
from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore

class DNSMOS:
    def __init__(self, sr=16000, use_gpu=True):
        """
        DNSMOS P.835 Scorer with Automatic GPU/CPU Fallback.

        Args:
            sr (int): Sampling rate (default: 16000).
            use_gpu (bool): Whether to attempt using GPU if available (default: True).
        """
        self.sr = sr
        self.device = "cpu"
        self.metric = None

        # Attempt to initialize with GPU first if requested
        if use_gpu and torch.cuda.is_available():
            try:
                self.metric = DeepNoiseSuppressionMeanOpinionScore(fs=sr, personalized=False, device="cuda")
                self.device = "cuda"
                print("✅ DNSMOS initialized on CUDA.")
            except Exception as e:
                print(f"⚠️  GPU initialization failed ({e}), falling back to CPU.")
        
        # Fallback to CPU if GPU was not used or failed
        if self.metric is None:
            self.metric = DeepNoiseSuppressionMeanOpinionScore(fs=sr, personalized=False, device="cpu")
            self.device = "cpu"
            print("✅ DNSMOS initialized on CPU.")

    def score(self, gen_wav: np.ndarray) -> dict:
        """
        Calculate DNSMOS P.835 scores for a single waveform.

        Args:
            gen_wav (np.ndarray): Generated waveform (T,).

        Returns:
            dict: Dictionary containing 'SIG', 'BAK', and 'OVRL' scores.
        """
        if gen_wav.ndim != 1:
            raise ValueError("Input waveform must be a 1D numpy array.")
        tensor = torch.from_numpy(gen_wav).float().to(self.device)
        scores = self.metric(tensor)
        return {"SIG": scores[1].item(), "BAK": scores[2].item(), "OVRL": scores[3].item()}

    def score_batch(self, wav_list: list) -> list:
        """
        Batch evaluation for a list of waveforms.

        Args:
            wav_list (list): List of numpy arrays (each with shape (T,)).

        Returns:
            list: List of dictionaries with 'SIG', 'BAK', and 'OVRL' scores.
        """
        results = []
        for idx, wav in enumerate(wav_list, start=1):
            print(f"Processing file {idx}/{len(wav_list)}...")
            try:
                result = self.score(wav)
            except Exception as e:
                print(f"❌ Failed to process file {idx}: {e}")
                result = {"SIG": None, "BAK": None, "OVRL": None}
            results.append(result)
        return results
