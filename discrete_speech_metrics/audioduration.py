class AudioDuration:
    def __init__(self, sr=16000):
        """
        Audio Duration scorer.

        Args:
            sr (int): Sampling rate (default: 16000).
        """
        self.sr = sr

    def score(self, ref_audio, gen_audio):
        """
        Calculate duration percentage error between reference and generated audio.

        Args:
            ref_audio (np.ndarray): Reference waveform (T,).
            gen_audio (np.ndarray): Generated waveform (T,).

        Returns:
            float: Percentage duration error relative to reference.
        """
        ref_duration_sec = len(ref_audio) / self.sr
        gen_duration_sec = len(gen_audio) / self.sr
        error_sec = abs(ref_duration_sec - gen_duration_sec)
        error_pct = (error_sec / ref_duration_sec * 100) if ref_duration_sec != 0 else 0.0
        return error_pct
