import os
import torch
from typing import Literal, List
from ..utils.model_loader import load_nemo_asr_model
from ..utils.constants import ASR_ALLOWED_LANGUAGES

PARAKEET_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
PARAKEET_MODEL, _, DEVICE = load_nemo_asr_model(PARAKEET_MODEL_NAME)

if PARAKEET_MODEL is None:
    print("Model loading failed. Terminating process.")
    exit()
else:
    print(f"✅ NeMo {PARAKEET_MODEL_NAME} has been loaded and is in use in {DEVICE}")

def transcribe_audio_parakeet(
        audio_path: str,
        target_language: ASR_ALLOWED_LANGUAGES
) -> str:
    """
    Creates a Transcription with Parakeet-tdt-0.6b-v3 (NeMo version).
    Parakeet recognises the language automatically from 25 european languages.

    :param audio_path: Path to audio file (recommended: WAV, 16 kHz, mono)
    :param target_language: Target language (used as metadata)
    :return: Transcription
    """

    if PARAKEET_MODEL is None:
        return "ERROR: No model loaded."

    if not os.path.exists(audio_path):
        print(f"ERROR: File '{audio_path}' not found.")
        return ""

    try:
        with torch.no_grad():
            transcriptions: List[str] = PARAKEET_MODEL.transcribe(
                paths2audio_files=[audio_path],
                batch_size=1,
                return_hypotheses=False
            )

        if transcriptions:
            return transcriptions[0]
        else:
            return "Transcription failed (empty result)."

    except Exception as e:
        print(f"ERROR running NeMo model: {e}")
        return ""