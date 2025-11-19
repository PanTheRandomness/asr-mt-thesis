import sys
import os
import subprocess
import argparse
import glob
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List
from utils.constants import SHORT_LANG_CODES

ASR_MODELS = {
    "whisper_medium": "whisper-medium",
    "wav2vec2_xlsr_fr": "wav2vec2-large-xlsr-53-french",
    "wav2vec2_xlsr_fi": "wav2vec2-large-xlsr-53-finnish",
    "wav2vec2_xlsr_en": "wav2vec2-large-960h",
    "parakeet": "parakeet-tdt-0.6b-v3",
}

MT_MODELS = {
    "nllb": "nllb-200-distilled-600M",
    "opus_mt_fi_en": "opus-mt-fi-en",
    "opus_mt_en_fi": "opus-mt-en-fi",
    "opus_mt_fi_fr": "opus-mt-fi-fr",
    "opus_mt_fr_fi": "opus-mt-tc-bible-big-deu_eng_fra_por_spa-fiu",
    "opus_mt_en_fr": "opus-mt-en-fr",
    "opus_mt_fr_en": "opus-mt-fr-en"
}


# TODO: Add batch evaluation script for each language & all transcriptions/translations!
def run_asr_evaluation(lang_code: str):
    """Runs ASR evaluation for all models for one language."""

def run_mt_evaluation(src_lang: str, tgt_lang: str):
    """Runs MT evaluation for all models for a single language pair."""

def compile_results():
    """Reads all JSON results and compiles them in one CSV file."""

def main():
    """Run all evaluations. """

if __name__ == "__main__":
    main()