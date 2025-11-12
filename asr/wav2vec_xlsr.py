import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import librosa
import torch

from utils.model_loader import load_wav2vec2_asr_model
from utils.constants import ASR_ALLOWED_LANGUAGES

WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
WAV2VEC2_MODEL, WAV2VEC2_PROCESSOR, DEVICE = load_wav2vec2_asr_model(WAV2VEC2_MODEL_NAME)

if WAV2VEC2_MODEL is None:
    print("Model loading failed. Terminating process.")
    exit()
else:
    print(f"✅ {WAV2VEC2_MODEL_NAME} has been loaded and is in use in {DEVICE}")

def transcribe_audio_wav2vec2(
        audio_path: str
) -> str:
    """
    Creates a Transcription with Wav2Vec 2.0 XLSR-53 (CTC).

    :param audio_path: Path to audio file (16 kHz, mono)
    :param target_language: Target language (used as metadata)
    :return: Transcription
    """

    if not os.path.exists(audio_path):
        print(f"ERROR: File '{audio_path}' not found.")
        return ""

    try:
        audio_input, _ = librosa.load(audio_path, sr=16000)
    except Exception as e:
        print(f"ERROR in audio processing: {e}")
        return ""

    input_values = WAV2VEC2_PROCESSOR(
        audio_input,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_values

    if DEVICE.startswith("cuda"):
        input_values = input_values.to(DEVICE)

    try:
        with torch.no_grad():
            logits = WAV2VEC2_MODEL(input_values).logits

        predicted_ids = torch.argmax(logits, dim=1)

        transcription = WAV2VEC2_PROCESSOR.batch_decode(predicted_ids)[0]
        return transcription.lower()

    except Exception as e:
        print(f"ERROR running NeMo model: {e}")
        return ""