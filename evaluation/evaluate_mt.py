import sys
import os
import re
import argparse
import json

import numpy as np
import pandas as pd

from evaluate import load
from typing import List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SHORT_LANG_CODES = ["fi", "en", "fr"]

def load_data(file_path: str) -> List[str]:
    """
    Load from text file (one sentence per line).

    :param file_path: Path to text file
    :return: Source text list
    """
    try:
        full_path = os.path.abspath(file_path)

        with open(full_path, 'r', encoding='utf-8') as f:
            source_texts = [line.strip() for line in f.readlines() if line.strip()]
        print(f"[Data Handler] Source texts loaded: {len(source_texts)} from file: {file_path}")
        return source_texts
    except FileNotFoundError:
        print(f"❌ ERROR: Fine {full_path} not found.")
        return []
    except Exception as e:
        print(f"❌ ERROR loading data from {full_path}: {e}")
        return []


def normalize(text: str) -> str:
    """
    Applies standard normalisation rules for metrics calculation:
    1. Lowercases the text.
    2. Removes all specified punctuation.
    3. Standardises all whitespace (including newlines) to a single space, effectively
        joining any pre-split sentences into a single, clean line for evaluation.
    """
    text = text.lower()
    text = re.sub(r'[.,:;!?"\'\\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = " ".join(text.split())
    return text

def calculate_mt_metrics(
        sources: List[str],
        predictions: List[str],
        references: List[List[str]],
        src_lang: str,
        tgt_lang: str
) -> Dict[str, float]:
    """
    Calculates MT-metrics (METEOR, TER, COMET)

    :param sources: List of source texts.
    :param predictions: List of predicted/translated texts.
    :param references: List of reference texts (List of "List of str").
    :param src_lang: Source language code (e.g., 'fi').
    :param tgt_lang: Target language code (e.g., 'en').
    :return: Dictionary containing the calculated metrics.
    """
    results = {}

    print("⏳ Normalizing texts for MT metrics...")
    try:
        sources = [normalize(s) for s in sources]
        predictions = [normalize(p) for p in predictions]
        references = [normalize(r) for r in references]
        print("✅ Normalization complete.")
    except Exception as e:
        print(f"❌ Normalization failed: {e}")
        sources = []
        predictions = []
        references = []

    if not sources or not predictions or not references:
        print("❌ ERROR: Source, predictions, or references list is empty.")
        return {"METEOR": float('nan'), "TER": float('nan'), "COMET": float('nan')}

    single_ref_list = [[ref] for ref in references]

    min_len = min(len(sources), len(predictions), len(single_ref_list))
    sources = sources[:min_len]
    predictions = predictions[:min_len]
    single_ref_list = single_ref_list[:min_len]

    if min_len < len(predictions) or min_len < len(references) or min_len < len(sources):
        print(f"⚠️ WARNING: Text lengths do not match. Used shortest length: {min_len}.")

    # METEOR
    print("⏳ Calculating METEOR...")
    try:
        meteor = load("meteor")
        results_meteor = meteor.compute(predictions=predictions, references=single_ref_list)
        results["METEOR"] = results_meteor.get('meteor', float('nan'))
        print(f"✅ METEOR calculated: {results['METEOR']:.4f}")
    except Exception as e:
        print(f"❌ METEOR loading/calculation failed: {e}")
        results["METEOR"] = float('nan')

    # TER
    print("⏳ Calculating TER...")
    try:
        ter = load("ter")
        results_ter = ter.compute(predictions=predictions, references=single_ref_list)
        results["TER"] = results_ter.get('score', float('nan'))
        print(f"✅ TER calculated: {results['TER']:.4f}")
    except Exception as e:
        print(f"❌ TER loading/calculation failed: {e}")
        results["TER"] = float('nan')

    # COMET
    print("⏳ Calculating COMET...")
    try:
        comet = load("comet", 'wmt20-comet-da',keep_in_memory=True)
        comet_results = comet.compute(
            predictions=predictions,
            references=[ref[0] for ref in single_ref_list],
            sources=sources
        )
        results["COMET"] = comet_results.get('mean_score', float('nan'))
        print(f"✅ COMET calculated (mean_score): {results['COMET']:.4f}")

    except Exception as e:
        print(
            f"❌ COMET loading/calculation failed (Check 'unbabel-comet' installation & GPU/memory capacity): {e}")
        results["COMET"] = float('nan')

    return results


def get_asr_metadata_and_metrics(asr_eval_file: str) -> Dict:
    """Reads ASR evaluation JSON and extracts relevant metadata and mean WER/WIL."""

    nan_metadata = {
        "WER": np.nan, "WIL": np.nan, "High_WER_Percentage": np.nan,
        "Speaker_Group": np.nan, "Speaker_Type": np.nan, "Speaker_Gender": np.nan,
        "Condition_Code": np.nan, "Noise_Level": np.nan, "Noise_Type": np.nan,
        "Voice_Volume": np.nan, "Audio_File_Name": np.nan
    }

    if asr_eval_file is None or not asr_eval_file or asr_eval_file.upper() == "HUMAN SOURCE":
        print(f"INFO: ASR Source status is {asr_eval_file}. Returning empty metadata structure.")

        dummy_data = {
            "Audio_File_Name": ["Human Source"],
            "Line_Index": [-1],
            "Speaker_Group": ["N/A"],
            "Speaker_Type": ["N/A"],
            "Speaker_Gender": ["N/A"],
            "Condition_Code": ["N/A"],
            "Noise_Level": ["N/A"],
            "Noise_Type": ["N/A"],
            "Voice_Volume": ["N/A"],
            "WER": [0.0],
            "WIL": [0.0],
            "High_WER_Percentage": [0.0],
        }
        return pd.DataFrame(dummy_data)

    try:
        asr_df = pd.read_json(asr_eval_file)

        if asr_df.empty:
            return nan_metadata

        first_row = asr_df.iloc[0].to_dict()

        meta_keys = [
            'Speaker_Group', 'Speaker_Type', 'Speaker_Gender',
            'Condition_Code', 'Noise_Level', 'Noise_Type', 'Voice_Volume',
            'Audio_File_Name'
        ]

        metadata = {k: first_row.get(k, np.nan) for k in meta_keys}

        mean_wer = asr_df['WER'].mean()
        mean_wil = asr_df['WIL'].mean()

        high_wer_percentage = (asr_df['WER'] > 0.4).sum() / len(asr_df) if len(asr_df) > 0 else np.nan

        metadata.update({
            "WER": mean_wer,
            "WIL": mean_wil,
            "High_WER_Percentage": high_wer_percentage
        })

        return metadata

    except Exception as e:
        print(f"❌ ERROR reading or parsing ASR evaluation JSON {asr_eval_file}: {e}")
        return nan_metadata

def save_evaluation_results(results_df: pd.DataFrame, output_path: str):
    """
    Saves results to JSON & CSV files.
    """

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = output_path.replace(".csv", ".json")
    try:
        results_df.to_json(json_path, orient='records', indent=4)
        print(f"✅ Results saved to JSON: {json_path}")
    except Exception as e:
        print(f"❌ ERROR saving JSON: {e}")

    # CSV
    csv_path = output_path.replace(".json", ".csv")
    try:
        results_df.to_csv(csv_path, index=False)
        print(f"✅ Results saved to CSV: {csv_path}")
    except Exception as e:
        print(f"❌ ERROR saving CSV: {e}")


def evaluate_single_translation():
    parser = argparse.ArgumentParser(
        description="Calculates MT metrics (METEOR, TER, COMET).",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", type=str, required=True, help="Model short name (e.g., nllb, opus).")
    parser.add_argument("--src_lang", type=str, required=True, choices=SHORT_LANG_CODES,
                        help="Source language (fi, en, fr).")
    parser.add_argument("--tgt_lang", type=str, required=True, choices=SHORT_LANG_CODES,
                        help="Target language (fi, en, fr).")
    parser.add_argument("--src_file", type=str, required=True,
                        help="Path to source file (e.g., data/results/mt/opus-mt-en-fi/translated_text.txt).")
    parser.add_argument("--ref_file", type=str, required=True,
                        help="Path to reference file (e.g., data/fi/fi_SOURCE.txt).")
    parser.add_argument("--pred_file", type=str, required=True,
                        help="Path to prediction file (e.g., data/results/mt/nllb-200-distilled-600M/fi_SOURCE_nllb_fi2en_results.txt).")
    parser.add_argument("--asr_wer_status", type=str, default="Human Source",
                        help="Status flag indicating the quality of the source transcription (e.g., 'WER_HIGH').")
    args = parser.parse_args()

    model_name = args.model
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    src_file = args.src_file
    ref_file = args.ref_file
    pred_file = args.pred_file
    asr_wer_status = args.asr_wer_status

    if asr_wer_status == "Human Source":
        asr_metadata = get_asr_metadata_and_metrics(None)
        asr_status_display = "Human Source"
    else:
        asr_metadata = get_asr_metadata_and_metrics(asr_wer_status)
        asr_status_display = f"ASR Source ({os.path.basename(asr_wer_status).replace('asr_eval_', '').replace('.json', '')})"

    print(f"\n--- MT evaluation: {model_name.upper()} ({src_lang.upper()} -> {tgt_lang.upper()}) ---")
    if asr_wer_status != "Human Source":
        print(f"⚠️ ASR WER Status: {asr_status_display} (Mean WER: {asr_metadata['WER']:.4f})")

    sources = load_data(src_file)
    references = load_data(ref_file)
    predictions = load_data(pred_file)

    if not sources or not references or not predictions:
        return

    metrics = calculate_mt_metrics(sources, predictions, references, src_lang, tgt_lang)

    results_data = {
        "Task": ["MT"],
        "Model": [model_name],
        "Language": [tgt_lang],
        "Audio_File_Name": [asr_metadata["Audio_File_Name"]],
        "Speaker_Group": [asr_metadata["Speaker_Group"]],
        "Speaker_Type": [asr_metadata["Speaker_Type"]],
        "Speaker_Gender": [asr_metadata["Speaker_Gender"]],
        "Condition_Code": [asr_metadata["Condition_Code"]],
        "Noise_Level": [asr_metadata["Noise_Level"]],
        "Noise_Type": [asr_metadata["Noise_Type"]],
        "Voice_Volume": [asr_metadata["Voice_Volume"]],
        "Source_Lang": [src_lang],
        "Target_Lang": [tgt_lang],
        "ASR_WER_Status": [asr_status_display],
        "High_WER_Percentage": [asr_metadata["High_WER_Percentage"]],
        "WER": [asr_metadata['WER']],
        "WIL": [asr_metadata['WIL']],
        "METEOR": [metrics['METEOR']],
        "TER": [metrics['TER']],
        "COMET": [metrics['COMET']],
        "Source_File": [src_file],
        "Ref_File": [ref_file],
        "Pred_File": [pred_file]
    }

    results_df = pd.DataFrame(results_data)

    print("\n✅ MT metrics calculated:")
    print(results_df[["METEOR", "TER", "COMET"]].to_markdown(index=False))

    output_filename = f"mt_evaluation_{model_name}_{src_lang}2{tgt_lang}.json"
    output_path = os.path.join("data", "results", "evaluation", "mt", output_filename)

    save_evaluation_results(results_df, output_path)

if __name__ == "__main__":
    evaluate_single_translation()