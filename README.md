# Discrete Speech Metrics++

Reference-aware automatic speech evaluation toolkit using self-supervised speech representations. 

## Install

To use `discrete-speech-metrics`, run the following: 
```bash
git clone https://github.com/nonmetal/DiscreteSpeechMetrics.git
cd DiscreteSpeechMetrics
python -m venv venv
source venv/bin/activate # Linux
# venv\Scripts\activate # Windows

pip3 install https://github.com/vBaiCai/python-pesq/archive/master.zip
pip3 install .
```

## Usage

### **1. Compute Speech Metrics**
```bash
python inference.py {reference_wav} {output_wav}
```

#### **Output Example**
```
Final Results:
SpeechBERTScore - Mean: 0.8069, Std: 0.0301
SpeechBLEU - Mean: 0.4982, Std: 0.0596
SpeechTokenDistance - Mean: 0.5372, Std: 0.0464
MCD - Mean: 7.0716, Std: 1.6591
Log F0 RMSE - Mean: 0.3317, Std: 0.1084
UTMOS - Mean: 1.9620, Std: 0.3617
```

---

### **2. Compute Error Rate**
The `error_rate.py` script calculates:
- **WER (Word Error Rate)** for English transcriptions.
- **CER (Character Error Rate)** for Japanese transcriptions (**use `--ja` flag**).

```bash
python error_rate.py {wav_path} {reference_script.json} [--ja]
```

#### **Expected Output**
```
english_1.wav | WER: 0.1352
english_2.wav | WER: 0.0784
...
Final WER Results:
Mean WER: 0.1543
```
or for **Japanese**:
```
japanese_1.wav | CER: 0.0812
japanese_2.wav | CER: 0.1243
...
Final CER Results:
Mean CER: 0.1027
```
The ground-truth transcripts in JSON format should follow this structure:
Each key **must match the corresponding WAV filename (without `.wav`)**.

#### **For English (WER)**
```json
{
    "english_sample": {"text": "Hello, this is a test sentence."},
    "japanese_sample": {"text": "このシステムは日本語の処理もできます."},
}
```

---

## References
- https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics
- https://github.com/wjassim/WARP-Q/tree/main/legacy_code
- https://github.com/dynamic-superb/espnet-whisper
```