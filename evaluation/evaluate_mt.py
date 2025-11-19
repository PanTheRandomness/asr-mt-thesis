import sys
import os
import argparse
import pandas as pd

from evaluate import load
from typing import List, Dict
from utils.constants import SHORT_LANG_CODES
from utils.data_handler import load_data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# TODO: Add COMET dependencies (loading may need its own script or evaluate.load('comet'))

def calculate_mt_metrics(sources: List[str], predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    """
    Calculates MT-metrics (METEOR, TER, COMET, MEANT)
    """
    results = {}

    if not sources or not predictions or not references:
        print("❌ ERROR: Source, predictions, or references list is empty.")
        return {"METEOR": float('nan'), "TER": float('nan'), "COMET": float('nan'), "MEANT": float('nan')}

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
        print(f"✅ METEOR calculated")
    except Exception as e:
        print(f"❌ METEOR loading/calculation failed: {e}")
        results["METEOR"] = float('nan')

    # TER
    print("⏳ Calculating TER...")
    try:
        ter = load("ter")
        results_ter = ter.compute(predictions=predictions, references=single_ref_list)
        results["TER"] = results_ter.get('score', float('nan'))
        print(f"✅ TER calculated")
    except Exception as e:
        print(f"❌ TER loading/calculation failed: {e}")
        results["TER"] = float('nan')

    # COMET
    print("⏳ Calculating COMET...")
    try:
        comet = load("comet", 'wmt20-comet-da')
        comet_results = comet.compute(
            predictions=predictions,
            references=single_ref_list,
            sources=sources
        )
        results["COMET"] = comet_results.get('mean_score', float('nan'))
        print(f"✅ COMET calculated (mean_score): {results['COMET']:.4f}")

    except Exception as e:
        print(
            f"❌ COMET loading/calculation failed (Check 'unbabel-comet' installation & GPU/memory capacity): {e}")
        results["COMET"] = float('nan')

    # TODO: MEANT
    # MEANT
    print("⏳ MEANT...")

    return results

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
        description="Calculates MT metrics (METEOR, TER, COMET, MEANT).",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", type=str, required=True, help="Model short name (e.g., nllb, opus-mt-fi-en).")
    parser.add_argument("--src_lang", type=str, required=True, choices=SHORT_LANG_CODES,
                        help="Source language (fi, en, fr).")
    parser.add_argument("--tgt_lang", type=str, required=True, choices=SHORT_LANG_CODES,
                        help="Target language (fi, en, fr).")
    parser.add_argument("--src_file", type=str, required=True, # TODO: why source & pred separately?
                        help="Path to source file (e.g., data/results/mt/opus-mt-en-fi/translated_text.txt).")
    parser.add_argument("--ref_file", type=str, required=True,
                        help="Path to reference file (e.g., data/fi/fi_SOURCE.txt).")
    parser.add_argument("--pred_file", type=str, required=True,
                        help="Path to prediction file (e.g., data/results/mt/nllb-200-distilled-600M/fi_SOURCE_nllb_fi2en_results.txt).")
    args = parser.parse_args()

    model_name = args.model
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    src_file = args.src_file
    ref_file = args.ref_file
    pred_file = args.pred_file

    print(f"\n--- MT evaluation: {model_name.upper()} ({src_lang.upper()} -> {tgt_lang.upper()}) ---")

    sources = load_data(src_file)
    references = load_data(ref_file)
    predictions = load_data(pred_file)

    if not sources or not references or not predictions:
        return

    metrics = calculate_mt_metrics(sources, predictions, references)

    results_data = {
        "Model": [model_name],
        "Source_Lang": [src_lang],
        "Target_Lang": [tgt_lang],
        "METEOR": [metrics['METEOR']],
        "TER": [metrics['TER']],
        "COMET": [metrics['COMET']],
        "MEANT": [metrics['MEANT']],
        "Source_File": [src_file],
        "Ref_File": [ref_file],
        "Pred_File": [pred_file]
    }

    results_df = pd.DataFrame(results_data)

    print("\n✅ MT metrics calculated:")
    print(results_df[["METEOR", "TER", "COMET", "MEANT"]].to_markdown(index=False))

    output_filename = f"mt_evaluation_{model_name}_{src_lang}2{tgt_lang}.json"
    output_path = os.path.join("data", "results", "evaluation", "mt", output_filename)

    save_evaluation_results(results_df, output_path)