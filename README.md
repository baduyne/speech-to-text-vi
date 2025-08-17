# Vietnamese Automatic Speech Recognition (ASR) System

## Introduction
This project implements an **Automatic Speech Recognition (ASR)** system for Vietnamese, with **multilingual support**.  
The system has been **fine-tuned** to reduce the Word Error Rate (WER) and integrates **environmental noise reduction** to improve recognition accuracy.  

The fine-tuned model is available at [baduyne/whisper-small-vi](https://huggingface.co/baduyne/whisper-small-vi).

## Key Features
- **Accurate Vietnamese speech recognition**, with multilingual support.  
- **Reduced WER** through fine-tuning of the base [whisper-small](https://huggingface.co/openai/whisper-small) model.  
- **Noise reduction** for environmental sounds such as fans, background chatter, etc.  
- **Cross-platform support**: Mobile, Windows, MacOS via format conversion from [whisper.cpp](https://github.com/ggml-org/whisper.cpp).  
- **Fast and lightweight** thanks to ggml optimizations and Q5 quantization, reducing the model size by ~5x (from ~1GB to just 190MB).
  
## Installation & Usage

### Install dependencies
```bash
pip install -r requirements.txt
```

Download resources
Since GitHub does not allow uploading files larger than 100MB, use the following script to download required resources:
```bash
python3 setup.py
```
Run API
```bash
uvicorn app:app --reload
```
Access the API
Open your browser and go to:
```bash
http://localhost:8000
```
### Test the model with different languages
German:
```bash
./whisper.cpp/build/bin/whisper-cli -m ggml-model-q5.ggml -f test_audio/de_test.mp3 -l auto -otxt
```
English:
```bash
./whisper.cpp/build/bin/whisper-cli -m ggml-model-q5.ggml -f test_audio/en_test.mp3 -l auto -otxt
```
- The -l auto option enables automatic language detection.
## References
- [Whisper.cpp](https://github.com/ggml-org/whisper.cpp) – Optimized Whisper implementation for CPU and cross-platform usage.
- [noisereduce](https://pypi.org/project/noisereduce/1.0.1/) – A Python library for environmental noise reduction.

