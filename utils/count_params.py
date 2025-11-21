import os
import sys
import torch
from typing import Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.constants import OPUS_MODEL_MAP, WAV2VEC2_MODEL_MAP
from utils.model_loader import load_wav2vec2_asr_model, load_opus_mt_model

def count_model_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_size(model_name: str, model: Any, task: str):

    if model is None:
        print(f"❌ Model loading failed: {model_name}. Skipping counting.")

    num_params = count_model_parameters(model)
    params_in_M = num_params / 1_000_000

    print(f"| {task:<3} | {model_name:<40} | {num_params:,} ({params_in_M:.2f}M) |")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_parameter_counting():
    print("=" * 80)
    print("✨ MODEL PARAMETER COUNTER✨")
    print("-" * 80)
    print(f"| {'Task':<3} | {'Model ID':<40} | {'Parameter count (M)':<31} |")
    print("=" * 80)

    print("\n--- ASR: Wav2Vec 2.0 (Quantised 4-bit) ---")
    for lang, model_id in WAV2VEC2_MODEL_MAP.items():
        model, _, _ = load_wav2vec2_asr_model(model_id)
        print_model_size(f"Wav2Vec2-{lang}", model, "ASR")

    print("\n--- MT: Helsinki-NLP Opus-MT (Float16) ---")
    for (src, tgt), model_id in OPUS_MODEL_MAP.items():
        model, _, _ = load_opus_mt_model(model_id)
        print_model_size(f"Opus-{src}2{tgt}", model, "MT")


    print("=" * 80)
    print("✅ Counting complete.")
    print("=" * 80)

if __name__ == "__main__":
    run_parameter_counting()