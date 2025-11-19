import sys
import os
import argparse
import pandas as pd
import jiwer

from typing import List, Dict
from utils.constants import SHORT_LANG_CODES, CONDITIONS_MAP
from utils.data_handler import load_data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def calculate_asr_metrics(references: List[str], predictions: List[str]) -> Dict [str, float]:
    """
    Calculates ASR-metrics.
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
        print(f"❌ ERROR saving to JSON: {e}")

    # CSV
    csv_path = output_path.replace(".json", ".csv")
    try:
        results_df.to_csv(csv_path, index=False)
        print(f"✅ Results saved to CSV: {csv_path}")
    except Exception as e:
        print(f"❌ ERROR saving to CSV: {e}")

def evaluate_single_transcription():
    parser = argparse.ArgumentParser(
        description="Calculates ASR metrics (WER, WIL).",
        fromfile_prefix_chars=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", type=str, required=True, help="Model short name (e.g., whisper-large).")
    parser.add_argument("--lang", type=str, required=True, choices=SHORT_LANG_CODES, help="Language (fi, en, fr).")
    parser.add_argument("--ref_file", type=str, required=True,
                        help="Path to reference file (e.g,. data/fi/fi_SOURCE.txt).")
    parser.add_argument("--pred_file", type=str, required=True,
                        help="Path to prediction file (e.g., data/results/asr/openai_whisper_large_v2/en_nat_m_c1_transcription.txt).")
    args = parser.parse_args()

    model_name = args.model
    lang_code = args.lang
    ref_file = args.ref_file
    pred_file = args.pred_file

    print(f"\n--- ASR evaluation: {model_name.upper()} ({lang_code.upper()}) ---")

    references = load_data(ref_file)
    predictions = load_data(pred_file)

    if not references or not predictions:
        return

    metrics = calculate_asr_metrics(references, predictions)

    # TODO: Add conditions
    results_data = {
        "Model": [model_name],
        "Language": [lang_code],
        "WER": [metrics['WER']],
        "WIL": [metrics['WIL']],
        "Ref_File": [ref_file],
        "Pred_file": [pred_file]
    }

    results_df = pd.DataFrame(results_data)

    print(f"\n✅ ASR metrics calculated for {pred_file}:")
    print(results_df[["WER", "WIL"]].to_markdown(index=False))

    output_filename = f"asr_eval_{model_name}_{lang_code}.json"
    output_path = os.path.join("data", "results", "evaluation", "asr", output_filename)

    save_evaluation_results(results_df, output_path)

if __name__ == "__main__":
    # TODO: Normalise!
    evaluate_single_transcription()