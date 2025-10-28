import os
from typing import List, Dict, Tuple
from evaluate import load # TODO: Fix cannot find
import jiwer
import csv
from utils.constants import SHORT_LANG_CODES, OPUS_MODEL_MAP

def load_texts(file_path: str) -> List[str]:
    """Loads texts from file (one sentence per line)."""
    try:
        full_path = os.path.abspath(file_path)
        with open(full_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"ERROR: Could not locate fine {file_path}")
        return []

def save_summary_to_csv(results: dict, filename: str = "data/results/summary/evaluation_summary.csv"):
    """Saves evaluation results to csv."""

    output_dir = os.path.dirname(filename)
    os.makedirs(output_dir, exist_ok=True)

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Task', 'Model', 'Language_Pair', 'Metric', 'Score' ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for key, metric in results.items():
            model_info, lang_pair = key.split('_', 1)
            task = 'ASR' if 'asr' in model_info.lower() or 'whisper' in model_info.lower() or 'wav2vec' in model_info.lower() or 'parakeet' in model_info.lower() else 'MT'

            for metric, score in metrics.items(): # TODO: Fix unresolved
                writer.writerow({
                    'Task': task,
                    'Model': model_info,
                    'Language_Pair': lang_pair.replace('2', '->'),
                    'Metric': metric,
                    'Score': score
                })
    print(f"\n✅ Evaluation Summary has been saved: {filename}.")

def calculate_asr_metrics(ground_truth: List[str], predictions: List[str]) -> Dict[str, float]:
    """Calculates ASR metrics (WER, WIL)."""

    metrics = jiwer.compute_measures(
        truth=ground_truth,
        hypothesis=predictions,
        truth_transform=jiwer.ToLowerCase().remove_punctuation(True).strip(True),
        hypothesis_transform=jiwer.ToLowerCase().remove_punctuation(True).strip(True)
    )

    return {
        "WER": round(metrics['wer'], 4),
        "WIL": round(metrics['wil'], 4),
        "MER": round(metrics['mer'], 4)
    }

def run_asr_evaluation(model_id: str, lang_code: str, gt_filename: str, pred_filename: str) -> Dict[str, float]:
    """Runs evaluation on ASR results"""
    model_safe_name = model_id.split("/")[-1].replace("-", "_").replace(".", "_")

    gt_path = os.path.join("data", "evaluation", "asr", lang_code, gt_filename)
    pred_path = os.path.join("data", "results", "asr", model_safe_name, lang_code, pred_filename)

    ground_truth = load_texts(gt_path)
    predictions = load_texts(pred_path)

    if not ground_truth or not predictions:
        print(f"[ASR] No data ({model_id} / {lang_code}). Skipping.")
        return {}

    min_len = min(len(ground_truth), len(predictions))
    if len(ground_truth) != len(predictions):
        print(f"WARNING: Sample size differs (GT: {len(ground_truth)}, Pred: {len(predictions)}). Using smaller {min_len} size.")
        ground_truth = ground_truth[:min_len]
        predictions = predictions[:min_len]

    metrics = calculate_asr_metrics(ground_truth, predictions)
    print(f"[ASR] ({model_safe_name} / {lang_code}) Results: {metrics}")
    return metrics

def calculate_mt_metrics(
        scr_texts: List[str],
        references: List[str],
        predictions: List[str],
        tgt_lang: str
) -> Dict[str, float]:
    """Calculates MT metrics (METEOR, TER, COMET, MEANT)."""

    references_for_mt = [[r] for r in references]
    metrics = {}

    # METEOR
    try:
        meteor = load("meteor")
        meteor_results = meteor.compute(predictions=predictions, references=references_for_mt, lang=tgt_lang)
        metrics["METEOR"] = round(meteor_results["meteor"], 4)
    except Exception as e:
        metrics["METEOR"] = float('nan')
        print(f"WARNING: METEOR-calculation failed: {e}.")

    # TER
    try:
        ter = load("ter")
        ter_results = ter.compute(predictiosn=predictions, references=references_for_mt)
        metrics["TER"] = round(ter_results["score"], 4)
    except Exception as e:
        metrics["TER"] = float('nan')
        print(f"WARNING: TER-calculation failed: {e}.")

    # COMET
    try:
        comet = load("comet", cache_dir="./.comet_cache")
        comet_results = comet.compute(
            predictions=predictions,
            references=references,
            sources=src_texts # TODO: Fix unresolved reference
        )
        metrics["COMET"] = round(comet_results["comet"], 4)
    except Exception as e:
        metrics["COMET"] = float('nan')
        print(f"WARNING: COMET-calculation failed: {e}.")

    # TODO: Add MEANT

def run_mt_evaluation(
        model_id: str,
        src_lang: str,
        tgt_lang: str,
        src_filename:str,
        ref_filename: str,
        pred_filename: str
) -> Dict[str, float]:
    """Runs MT evaluation."""

    model_safe_name = model_id.split("/")[-1].replace("-", "_")

    src_path = os.path.join("data", "source", "mt_texts", src_lang, src_filename)
    ref_path = os.path.join("data", "source", "mt", tgt_lang, ref_filename)
    pred_path = os.path.join("data", "results", "mt", model_safe_name, f"{src_lang}-{tgt_lang}", pred_filename)

    src_texts = load_texts(src_path)
    references = load_texts(ref_path)
    predictions = load_texts(pred_path)

    if not references or not predictions or not src_texts:
        print(f"[MT] No data ({model_id} / {src_lang} -> {tgt_lang}). Skipping.")
        return {}

    min_len = min(len(src_texts), len(references), len(predictions))
    if not (len(src_texts) == len(references) == len(predictions)):
        print(f"WARNING: Sample size differs (Src: {len(src_texts)}, Ref: {len(references)}, Pred: {len(predictions)})."
              f"Using smaller {min_len} size.")
        src_texts = src_texts[:min_len]
        references = references[:min_len]
        predictions = predictions[:min_len]

    metrics = calculate_mt_metrics(src_texts, references, predictions, tgt_lang)
    print(f"[MT] ({model_safe_name} / {src_lang} -> {tgt_lang}) Results: {metrics}")
    return metrics

ASR_MODELS = [
    "openai/whisper-medium",
    "facebook/wav2vec2-large-xlsr-53",
    "nvidia/parakeet-tdt-0.6b-v3"
]
# TODO: Add Opus
MT_MODEL_IDS = {
    "nllb": "facebook/nllb-200-distilled-600M",
    "bloom": "bigscience/bloom-560m"
}
LANG_PAIRS = [("fi", "en"), ("en", "fi"), ("fi", "fr"), ("fr", "fi"), ("en", "fr"), ("fr", "en")]

def main():
    print("=" * 50)
    print("Evaluation of Results")
    print("=" * 50)

    all_results = {}

    ASR_GT_FILE = "ground_truth.txt"
    ASR_PRED_FILE = "predictions.txt"
    MT_SRC_FILE = "source.txt"
    MT_REF_FILE = "reference.txt"

    print("\n[PART 1] ASR model evaluation")
    print("-" * 40)
    for model_id in ASR_MODELS:
        for lang in SHORT_LANG_CODES:
            key = f"{model_id}_{lang}"
            results = run_asr_evaluation(model_id, lang, ASR_GT_FILE, ASR_PRED_FILE)
            if results:
                all_results[key] = results

    print(f"[PART 2] MT model evaluation.")
    print("-" * 40)

    for model_alias, model_id in MT_MODEL_IDS.items():
        model_safe_name = model_id.split("/")[-1].replace("-", "_")
        for src, tgt in LANG_PAIRS:
            key = f"{model_id}_{src}2{tgt}"
            pred_file = run_mt_evaluation(model_id, src, tgt, MT_SRC_FILE, MT_REF_FILE, pred_file)
            if results:
                all_results[key] = results

    print("\n" + "=" * 50)
    print("Summary of the whole Evaluation: ")
    print("=" * 50)

    print("ASR results (Examples):")
    for key, metrics in all_results.items():
        if "whisper" in key.lower() and key.endswith("_fi"):
            print(f"  {key}: {metrics}")
            break

    print("\nMT results (Examples):")
    for key, metrics in all_results.items():
        if "nllb" in key.lower() and key.endswith("fi2en"):
            print(f"  {key}: {metrics}")
            break

    save_summary_to_csv(all_results)

if __name__ == "__main__":
    main()