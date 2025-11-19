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

SPEAKER_GROUPS = [
    "nat_m",
    "nat_n",
    "aks_m",
    "aks_n"
]

CONDITIONS_CODES = [f"c{i}" for i in range(1, 7)]

# TODO: Add batch evaluation script for each language & all transcriptions/translations!
def run_asr_evaluation(lang_code: str):
    """Runs ASR evaluation for all models for one language."""
    ref_file = os.path.join("data", lang_code, f"{lang_code}_SOURCE.txt")
    asr_script_path = os.path.join(os.path.dirname(__file__), "evaluate_asr.py")

    print(f"\n\n{'='*20} ASR EVALUATION ({lang_code.upper()}) {'='*20}")

    for short_model_name, folder_name in ASR_MODELS.items():
        print(f"\n--- Model: {short_model_name.upper()} ---")
        pred_file = os.path.join("data", "results", "asr", folder_name, f"{lang_code}_transcription.txt")

        if not os.path.exists(pred_file):
            print(f"❌ Skipping: Prediction file not found: {pred_file}")
            continue

        if not os.path.exists(ref_file):
            print(f"❌ Skipping: Reference file not found: {ref_file}")
            continue

        command = [
            sys.executable,
            asr_script_path,
            "--model", short_model_name,
            "--lang", lang_code,
            "--ref_file", ref_file,
            "--pred_file", pred_file
        ]

        try:
            subprocess.run(command, check=True, capture_output=False, text=True, cwd=os.getcwd())
            print(f"✅ ASR evaluation complete: {short_model_name} / {lang_code}")
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR evaluating ASR task {short_model_name} / {lang_code}: {e}")

def run_mt_evaluation(src_lang: str, tgt_lang: str):
    """Runs MT evaluation for all models for a single language pair."""

    base_source_file_name = f"{src_lang.upper()}_SOURCE.txt"
    base_file_name_no_ext = os.path.splitext(base_source_file_name)[0]

    src_file = os.path.join("data", src_lang, base_source_file_name) # TODO: Correct filenames
    ref_file = os.path.join("data", tgt_lang, f"{src_lang}_SOURCE.txt")

    mt_script_path = os.path.join(os.path.dirname(__file__), "evaluate_mt.py")

    print(f"\n\n{'=' * 20} MT EVALUATION ({src_lang.upper()} -> {tgt_lang.upper()}) {'=' * 20}")

    current_mt_models = {
        k: v for k, v in MT_MODELS.items()
        if f"{src_lang}_{tgt_lang}" in k or k.startswith("nllb")
    }

    for short_model_name, folder_name in current_mt_models.items():
        print(f"\n--- Malli: {short_model_name.upper()} ---")

        model_part = short_model_name.split("_")[0]
        pred_file = os.path.join(
            "data", "results", "mt",
            folder_name,
            f"{base_file_name_no_ext}_{model_part}_{src_lang}2{tgt_lang}_results.txt"
        )

        if not os.path.exists(ref_file):
            print(f"❌ Skipping: Source file not found: {ref_file}")
            continue

        if not os.path.exists(pred_file):
            print(f"❌ Skipping: Prediction file not found: {pred_file}")
            continue

        command = [
            sys.executable,
            mt_script_path,
            "--model", short_model_name,
            "--src_lang", src_lang,
            "--tgt_lang", tgt_lang,
            "--src_file", src_file,
            "--ref_file", ref_file,
            "--pred_file", pred_file
        ]

        try:
            subprocess.run(command, check=True, capture_output=False, text=True, cwd=os.getcwd())
            print(f"✅ MT evaluation complete: {short_model_name} / {src_lang} -> {tgt_lang}")
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR evaluating MT task {short_model_name} / {src_lang} -> {tgt_lang}: {e}")

def compile_results():
    """Reads all JSON results and compiles them in one CSV file."""

    output_dir = os.path.join("data", "results", "evaluation")
    asr_files = glob.glob(os.path.join(output_dir, "asr", "*.json"))
    mt_files = glob.glob(os.path.join(output_dir, "mt", "*.json"))

    all_files = asr_files + mt_files

    if not all_files:
        print("\n⚠️ Result JSON files not found.")
        return

    print(f"\n\n{'=' * 20} COMPILING RESULTS ({len(all_files)} files) {'=' * 20}")

    all_data = []

    for f in all_files:
        try:
            df = pd.read_json(f)
            df['Task'] = 'ASR' if 'asr_evaluation' in f else 'MT'
            all_data.append(df)
        except Exception as e:
            print(f"❌ ERROR reading file {f}: {e}")

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)

        cols = ['Task', 'Model', 'Language', 'Speaker_Group', 'Speaker_Type', 'Speaker_Gender',
                'Condition_Code', 'Noise_Level', 'Noise_Type', 'Voice_Volume',
                'Source_Lang', 'Target_Lang',
                'WER', 'WIL', 'METEOR', 'TER', 'COMET', 'MEANT']
        cols = [c for c in cols if c in final_df.columns]
        remaining_cols = [c for c in final_df.columns if c not in cols]
        final_df = final_df[cols + remaining_cols]

        final_csv_path = os.path.join(output_dir, "FINAL_RESULTS_SUMMARY.csv")
        final_json_path = os.path.join(output_dir, "FINAL_RESULTS_SUMMARY.json")

        final_df.to_csv(final_csv_path, index=False)
        final_df.to_json(final_json_path, orient='records', indent=4)

        print("\n✅ ALL RESULTS COMPILED:")
        print(f"CSV summary: {final_csv_path}")
        print(f"JSON summary: {final_json_path}")
    else:
        print("❌ No results to compile.")

def main():
    """Run all evaluations. """
    print("=" * 70)
    print(" STARTING EVALUATION OF ALL RESULTS ")
    print("=" * 70)

    for lang in SHORT_LANG_CODES:
        run_asr_evaluation(lang)

    for src_lang in SHORT_LANG_CODES:
        for tgt_lang in SHORT_LANG_CODES:
            if src_lang != tgt_lang:
                run_mt_evaluation(src_lang, tgt_lang)

    compile_results()

    print("=" * 70)
    print(" ALL EVALUATIONS COMPLETED ")
    print("=" * 70)

if __name__ == "__main__":
    main()