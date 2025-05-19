import torch
import numpy as np
from speechbrain.inference import EncoderClassifier
from scipy.spatial.distance import cosine

class SpeakerSimilarity:
    def __init__(self, use_gpu=True):
        """
        Speaker Similarity (SIM) scorer using ECAPA-TDNN (SpeechBrain 1.0+ API).
        """
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device}
        )

    def extract_embedding(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract speaker embedding from waveform.
        """
        tensor = torch.from_numpy(wav).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.encoder.encode_batch(tensor)  # Shape: (1, embedding_dim)
            embedding = embedding.squeeze(0).squeeze(0).cpu().numpy().flatten()  # Ensure 1D
        return embedding


    def compute_similarity(self, ref_wav: np.ndarray, gen_wav: np.ndarray) -> float:
        """
        Compute cosine similarity between reference and generated speech.
        """
        emb_ref = self.extract_embedding(ref_wav)
        emb_gen = self.extract_embedding(gen_wav)
        similarity = 1 - cosine(emb_ref, emb_gen)
        return similarity
