# Discrete Speech Metrics++

Reference-aware automatic speech evaluation toolkit using self-supervised speech representations. 

## Install

To use `discrete-speech-metrics`, run the following. 
```bash
git clone https://github.com/nonmetal/DiscreteSpeechMetrics.git
cd DiscreteSpeechMetrics
python -m venv venv
source venv/bin/activate #Linux
# venv\Scripts\activate # Windows

pip3 install https://github.com/vBaiCai/python-pesq/archive/master.zip
pip3 install .
```

## Usage

```bash
python inference.py --reference_wav --output_wav
```

## Output

```
Final Results:
SpeechBERTScore - Mean: 0.8069, Std: 0.0301
SpeechBLEU - Mean: 0.4982, Std: 0.0596
SpeechTokenDistance - Mean: 0.5372, Std: 0.0464
MCD - Mean: 7.0716, Std: 1.6591
Log F0 RMSE - Mean: 0.3317, Std: 0.1084
UTMOS - Mean: 1.9620, Std: 0.3617
```

## References
- https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics
- https://github.com/wjassim/WARP-Q/tree/main/legacy_code
- https://github.com/jitsi/jiwer
