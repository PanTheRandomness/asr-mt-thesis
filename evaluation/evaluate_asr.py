import sys
import os
import re
import argparse
import jiwer

import pandas as pd
import numpy as np

from typing import List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SHORT_LANG_CODES = ["fi", "en", "fr"]

CONDITIONS_MAP = {
    "c1": {
        "background_noise_level": "low/none",
        "background_noise_type": None,
        "voice_volume": "normal"
    },
    "c2": {
        "background_noise_level": "low/none",
        "background_noise_type": None,
        "voice_volume": "low"
    },
    "c3": {
        "background_noise_level": "medium",
        "background_noise_type": "human",
        "voice_volume": "normal"
    },
    "c4": {
        "background_noise_level": "medium",
        "background_noise_type": "traffic",
        "voice_volume": "normal"
    },
    "c5": {
        "background_noise_level": "high",
        "background_noise_type": "human",
        "voice_volume": "normal"
    },
    "c6": {
        "background_noise_level": "high",
        "background_noise_type": "traffic",
        "voice_volume": "normal"
    }
}

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

def calculate_asr_metrics(references: List[str], predictions: List[str]) -> Dict [str, float]:
    """
    Calculates ASR-metrics (WER, WIL) for normalised text.

    :param references: List of reference texts (must be normalised).
    :param predictions: List of predictions (must be normalised).
    :return: Calculated metrics.
    """
    if not references or not predictions:
        print("❌ ERROR: References or predictions list is empty.")
        return {"WER": float('nan'), "WIL": float('nan')}

    if len(references) != len(predictions):
        print(
            f"⚠️ WARNING: The count of references ({len(references)}) & predictions ({len(predictions)}) is not evenly matched.")
        min_len = min(len(references), len(predictions))
        references = references[:min_len]
        predictions = predictions[:min_len]

    metrics = jiwer.compute_measures(
        references,
        predictions
    )

    return {
        "WER": metrics['wer'],
        "WIL": metrics['wil']
    }

def load_and_aggregate_results(filepath: str, new_result: Dict) -> pd.DataFrame:
    """
    Loads existing results (JSON), appends the new single result, and returns the DataFrame.
    """
    try:
        if os.path.exists(filepath):
            df = pd.read_json(filepath)
        else:
            df = pd.DataFrame()

        new_df = pd.DataFrame([new_result])
        df = pd.concat([df, new_df], ignore_index=True)
        return df
    except Exception as e:
        print(f"⚠️ Warning: Could not load/aggregate ASR results from {filepath}. Starting new. Error: {e}")
        return pd.DataFrame([new_result])

def save_evaluation_results(results_df: pd.DataFrame, output_path: str):
    """
    Saves results to a single JSON file.
    Final aggregation to CSV takes place in run_evaluation.py.
    """

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        results_df.to_json(output_path, orient='records', indent=4)
    except Exception as e:
        print(f"❌ ERROR saving ASR JSON: {e}")

def evaluate_single_transcription():
    parser = argparse.ArgumentParser(
        description="Calculates ASR metrics (WER, WIL) for a single transcription file against a full reference file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", type=str, required=True, help="Model short name (e.g., whisper-large).")
    parser.add_argument("--lang", type=str, required=True, choices=SHORT_LANG_CODES, help="Language (fi, en, fr).")
    parser.add_argument("--ref_file", type=str, required=True,
                        help="Path to the FULL normalized reference file (e.g,. data/fi/fi_SOURCE.txt).")
    parser.add_argument("--pred_file", type=str, required=True,
                        help="Path to the single (ASR) prediction file (e.g., data/results/asr/openai_whisper_large_v2/en_nat_m_c1_transcription.txt).")
    parser.add_argument("--line_index", type=int, default=None,
                        help="The line index in the full reference file that corresponds to the prediction (0-indexed).")
    args = parser.parse_args()

    full_references = load_data(args.ref_file)
    predictions = load_data(args.pred_file)

    filename = os.path.basename(args.pred_file)
    match_model_suffix = re.search(r'_transcription\.txt$', filename, re.IGNORECASE)

    if match_model_suffix:
        # fi_nat_n_c4
        filename_base = filename[:match_model_suffix.start()]
    else:
        filename_base = os.path.splitext(filename)[0]

    audio_file_name = f"{filename_base}.wav"
    line_index = args.line_index

    if line_index is None:
        line_index = -1
        match_index = re.search(r'[-_]c(\d+)', filename_base, re.IGNORECASE)

        if match_index:
            try:
                index_str = match_index.group(1)
                line_index = int(index_str) - 1
            except ValueError:
                line_index = -1

    if len(full_references) == 1:
        line_index = 0

    if line_index < 0 or line_index >= len(full_references):
        print(f"❌ ERROR: Invalid or uninferrable line index ({line_index}) for file {audio_file_name}. Skipping.")
        return

    if len(predictions) == 0:
        print(f"❌ ERROR: Prediction file {args.pred_file} is empty.")
        return

    single_prediction_line = " ".join(predictions).strip()
    single_reference_line = " ".join(full_references).strip()

    print("⏳ Normalising predictions for ASR metrics...")
    predictions_normalized = [normalize(single_prediction_line)]
    references = [normalize(single_reference_line)]

    metrics = calculate_asr_metrics(references, predictions_normalized)

    if pd.isna(metrics['WER']):
        return

    try:
        match = re.match(r'([a-z]{2})[-_]([a-z]{3})[-_]([a-z]{1})[-_](c[1-6])', filename_base)

        if match:
            lang_code = match.group(1)
            speaker_type = match.group(2)
            speaker_gender = match.group(3)
            condition_code = match.group(4)
            speaker_group = f"{speaker_type}_{speaker_gender}"

            condition_data = CONDITIONS_MAP.get(condition_code, {})

            meta_data = {
                "Speaker_Group": speaker_group,
                "Speaker_Type": speaker_type,
                "Speaker_Gender": speaker_gender,
                "Condition_Code": condition_code,
                "Noise_Level": condition_data.get("background_noise_level", np.nan),
                "Noise_Type": condition_data.get("background_noise_type", np.nan),
                "Voice_Volume": condition_data.get("voice_volume", np.nan),
                "Source_Lang": args.lang,
                "Target_Lang": np.nan,
                "ASR_WER_Status": "N/A (ASR Task)",
                "High_WER_Percentage": np.nan,
            }

            audio_file_name = f"{lang_code}-{speaker_type}-{speaker_gender}-{condition_code}.wav"
        else:
            print(f"⚠️ Varoitus: Ei kyetty parsimaan metatietoja ASR-tiedostosta: {filename_base}")
            meta_data = {
                "Speaker_Group": np.nan, "Speaker_Type": np.nan, "Speaker_Gender": np.nan,
                "Condition_Code": np.nan, "Noise_Level": np.nan, "Noise_Type": np.nan,
                "Voice_Volume": np.nan, "Source_Lang": args.lang, "Target_Lang": np.nan,
                "ASR_WER_Status": "N/A (ASR Task)", "High_WER_Percentage": np.nan
            }

    except Exception as e:
        print(f"❌ Virhe ASR-metatietojen parsinnassa: {e}")
        meta_data = {
            "Speaker_Group": np.nan, "Speaker_Type": np.nan, "Speaker_Gender": np.nan,
            "Condition_Code": np.nan, "Noise_Level": np.nan, "Noise_Type": np.nan,
            "Voice_Volume": np.nan, "Source_Lang": args.lang, "Target_Lang": np.nan,
            "ASR_WER_Status": "N/A (ASR Task)", "High_WER_Percentage": np.nan
        }

    print(f"\n--- ASR Evaluation for {audio_file_name} (Line Index: {line_index}) ---")
    print(f"  WER: {metrics['WER']:.4f}")
    print(f"  WIL: {metrics['WIL']:.4f}")

    result_data = {
        "Task": "ASR",
        "Model": args.model,
        "Language": args.lang,
        "Audio_File_Name": audio_file_name,
        "Line_Index": line_index,
        "WER": metrics['WER'],
        "WIL": metrics['WIL'],
        "METEOR": np.nan,
        "TER": np.nan,
        "COMET": np.nan,
        **meta_data,
        "Ref_Text_Normalized": references[0],
        "Pred_Text_Normalized": predictions_normalized[0],
        "Ref_File_Path": args.ref_file,
        "Pred_File_Path": args.pred_file,
    }

    try:
        model_folder_name = args.pred_file.split(os.sep)[-2]
    except IndexError:
        model_folder_name = "unknown_model"
        print(f"⚠️ WARNING: Could not infer model folder name from path: {args.pred_file}")

    output_path = os.path.join("data", "results", "evaluation", "asr", f"asr_eval_{args.lang}_{model_folder_name}.json")

    current_df = load_and_aggregate_results(output_path, result_data)
    save_evaluation_results(current_df, output_path)

if __name__ == "__main__":
    evaluate_single_transcription()