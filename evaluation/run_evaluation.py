import sys
import os
import subprocess
import glob

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.constants import SHORT_LANG_CODES, ASR_MODELS, MT_MODELS
from utils.data_handler import load_data, write_data, normalize

PREDICTION_FILE_PATTERN = "*.txt"
NORMALIZED_DIR = os.path.join("data", "normalized")
MT_SOURCES_DIR = os.path.join("data", "results", "mt_sources")

SPEAKER_GROUPS = [
    "nat_m",
    "nat_n",
    "aks_m",
    "aks_n"
]

CONDITIONS_CODES = [f"c{i}" for i in range(1, 7)]

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

def normalize_all_references():
    """Loads, normalises and saves all SOURCE-references once."""

    os.makedirs(NORMALIZED_DIR, exist_ok=True)
    print("\n--- Normalising all reference texts once. ---")

    for lang_code in SHORT_LANG_CODES:
        raw_ref_path = os.path.join("data", lang_code, f"{lang_code}_SOURCE.txt")
        normalized_ref_path = os.path.join(NORMALIZED_DIR, f"{lang_code}_SOURCE.txt")

        if os.path.exists(normalized_ref_path):
            print(f"✅ Skipping {lang_code}: Normalized reference already exists at {normalized_ref_path}")
            continue

        if not os.path.exists(raw_ref_path):
            print(f"⚠️ Warning: Raw reference file not found for {lang_code}: {raw_ref_path}")
            continue

        try:
            raw_texts = load_data(raw_ref_path)

            normalized_texts = [normalize(text) for text in raw_texts]

            single_line = " ".join(normalized_texts).strip()

            if write_data([single_line], normalized_ref_path):
                print(f"✅ Normalized reference saved for {lang_code} to {normalized_ref_path}")

        except Exception as e:
            print(f"❌ ERROR processing {lang_code} reference: {e}")

def concatenate_asr_outputs_for_mt(lang_code: str, asr_folder_name: str) -> str:
    """
    Aggregates all single ASR transcriptions to a single file.
    Done so that MT-metrics such as COMET are able to calculate all source texts from a single file.
    """

    asr_eval_path = os.path.join("data", "results", "evaluation", "asr", f"asr_eval_{lang_code}_{asr_folder_name}.json")
    if not os.path.exists(asr_eval_path):
        print(f"❌ Cannot find ASR evaluation results for concatenation: {asr_eval_path}")
        return None

    aggregated_path = os.path.join(MT_SOURCES_DIR, f"{lang_code}_{asr_folder_name}_ASR_SOURCE.txt")

    try:
        asr_df = pd.read_json(asr_eval_path)
        asr_df = asr_df.sort_values(by='Line_Index')

        aggregated_texts = []

        for index, row in asr_df.iterrows():
            audio_filename_no_ext = os.path.splitext(row['Audio_File_Name'])[0]
            asr_file_path = os.path.join(
                "data", "results", "asr", asr_folder_name,
                f"{audio_filename_no_ext}_{asr_folder_name.replace('-', '_')}_transcription.txt"
            )

            if os.path.exists(asr_file_path):
                with open(asr_file_path, 'r', encoding='utf-8') as f:
                    transcription = f.read().strip().replace('\n', ' ')
                    aggregated_texts.append(transcription)
            else:
                aggregated_texts.append("")

        write_data(aggregated_texts, aggregated_path)

        return aggregated_path

    except Exception as e:
        print(f"❌ ERROR during ASR output concatenation for {lang_code}/{asr_folder_name}: {e}")
        return None

def run_asr_evaluation(lang_code: str):
    """Runs ASR evaluation for all models for one language."""
    ref_file = os.path.join(NORMALIZED_DIR, f"{lang_code}_SOURCE.txt")
    asr_script_path = os.path.join(os.path.dirname(__file__), "evaluate_asr.py")

    if not os.path.exists(ref_file):
        print(f"❌ Skipping language: Reference file not found: {ref_file}")
        return

    print(f"\n\n{'='*20} ASR EVALUATION ({lang_code.upper()}) {'='*20}")

    for short_model_name, folder_name in ASR_MODELS.items():
        if "wav2vec2" in short_model_name and not short_model_name.endswith(lang_code):
            continue

        output_path = os.path.join("data", "results", "evaluation", "asr", f"asr_eval_{lang_code}_{folder_name}.json")
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"INFO: Removed old ASR evaluation JSON: {output_path}")

        print(f"\n--- Model: {short_model_name.upper()} ---")

        model_result_dir = os.path.join("data", "results", "asr", folder_name)

        search_pattern_dash = os.path.join(model_result_dir, f"{lang_code}-*.txt")
        search_pattern_underscore = os.path.join(model_result_dir, f"{lang_code}_*.txt")

        pred_files = glob.glob(search_pattern_dash) + glob.glob(search_pattern_underscore)

        if not pred_files:
            print(f"⚠️ Warning: No prediction files found for {short_model_name}.")
            continue

        print(f"Found {len(pred_files)} transcription files to evaluate.")

        for pred_file in pred_files:

            command = [
                sys.executable,
                asr_script_path,
                "--model", short_model_name,
                "--lang", lang_code,
                "--ref_file", ref_file,
                "--pred_file", pred_file
            ]

            try:
                # Standardized subprocess call to match mt/run_all_mt.py
                subprocess.run(
                    command,
                    check=True,
                    text=True,
                    cwd=os.getcwd(),
                    stdout=sys.stdout,
                    stderr=sys.stderr
                )
                print(
                    f"  ✅ ASR evaluation complete for file {os.path.basename(pred_file)} (Results aggregated in: {os.path.basename(output_path)})")
            except subprocess.CalledProcessError as e:
                print(f"  ❌ ERROR evaluating ASR task for file {os.path.basename(pred_file)}!")
                print(f"  Command: {' '.join(e.cmd)}")
                stderr_content = e.stderr.strip() if e.stderr else "(No stderr output)"
                print(f"❌ Standard Error (Stderr): {stderr_content}")
                print("-------------------------------------------------")
                continue
            except FileNotFoundError:
                print(f"❌ FATAL ERROR: evaluate_asr.py script file not found.")
                raise

def run_mt_evaluation_on_human_source(src_lang: str, tgt_lang: str):
    """Runs MT evaluation using a manually written source."""

    base_source_file_name = f"{src_lang}_SOURCE.txt"
    base_file_name_no_ext = os.path.splitext(base_source_file_name)[0]

    src_file = os.path.join("data", src_lang, base_source_file_name)
    ref_file = os.path.join("data", tgt_lang, f"{tgt_lang}_SOURCE.txt")

    mt_script_path = os.path.join(os.path.dirname(__file__), "evaluate_mt.py")

    print(f"\n\n{'=' * 20} MT EVALUATION (HUMAN SOURCE) ({src_lang.upper()} -> {tgt_lang.upper()}) {'=' * 20}")

    current_mt_models = {
        k: v for k, v in MT_MODELS.items()
        if f"{src_lang}_{tgt_lang}" in k or k.startswith("nllb")
    }

    for short_model_name, folder_name in current_mt_models.items():
        print(f"\n--- Model: {short_model_name.upper()} (Human Source) ---")

        model_part = short_model_name.split("_")[0]

        human_source_folder = "test_translations"
        pred_file = os.path.join(
            "data", "results", "mt",
            human_source_folder,
            f"{base_file_name_no_ext}_{model_part}_{src_lang}2{tgt_lang}_results.txt"
        )

        if not os.path.exists(pred_file):
            print(f"❌ Skipping: Prediction file not found: {pred_file}. Run translation first.")
            continue

        command = [
            sys.executable,
            mt_script_path,
            "--model", f"{short_model_name}_human_source",
            "--src_lang", src_lang,
            "--tgt_lang", tgt_lang,
            "--src_file", src_file,
            "--ref_file", ref_file,
            "--pred_file", pred_file,
            "--asr_wer_status", "Human Source"
        ]

        try:
            subprocess.run(
                command,
                check=True,
                text=True,
                cwd=os.getcwd(),
                stdout=sys.stdout,
                stderr=sys.stderr
            )
            print(f"✅ MT evaluation complete: {short_model_name} / {src_lang} -> {tgt_lang} (Human Source)")
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR evaluating MT task {short_model_name} / {src_lang} -> {tgt_lang}!")
            print(f"Command: {' '.join(e.cmd)}")
            stderr_content = e.stderr.strip() if e.stderr else "(No stderr output)"
            print(f"❌ Standard Error (Stderr): {stderr_content}")
            print("-------------------------------------------------")
        except FileNotFoundError:
            print(f"❌ FATAL ERROR: evaluate_mt.py script file not found.")
            raise

def run_mt_evaluation_on_asr_source(src_lang: str, tgt_lang: str):
    """
    Evaluates MT performance when the source text is the ASR output, using
    the ASR evaluation JSON to flag the result based on source quality.
    """

    base_ref_file_name = f"{tgt_lang}_SOURCE.txt"
    ref_file = os.path.join("data", tgt_lang, base_ref_file_name)

    mt_script_path = os.path.join(os.path.dirname(__file__), "evaluate_mt.py")

    print(f"\n\n{'=' * 20} MT ON ASR SOURCE EVALUATION ({src_lang.upper()} -> {tgt_lang.upper()}) {'=' * 20}")

    for asr_short_name, asr_folder_name in ASR_MODELS.items():

        # Only let language-specific wav2vec models through, like "wav2vec2_xlsr_fi"
        if "wav2vec2" in asr_short_name and not asr_short_name.endswith(src_lang):
            continue

        print(f"\n--- ASR Source: {asr_short_name.upper()} ---")

        for speaker_group in SPEAKER_GROUPS:
            for condition_code in CONDITIONS_CODES:
                segment_prefix = f"{speaker_group.replace('_', '-')}-{condition_code}"
                if "_" in asr_folder_name and not asr_folder_name.startswith("opus"):
                    # Esim. nvidia_parakeet_tdt_0_6b_v3 -> parakeet_tdt_0_6b_v3 -> parakeet-tdt-0-6b-v3
                    asr_model_file_part = "_".join(asr_folder_name.split("_")[1:]).replace('_', '-')
                else:
                    asr_model_file_part = asr_folder_name.replace('_', '-')
                asr_output_base_name = f"{src_lang}_{speaker_group}_{condition_code}_{asr_folder_name}_transcription"
                asr_source_file = os.path.join(
                    "data", "results", "asr",
                    asr_folder_name,
                    f"{asr_output_base_name}.txt"
                )

                if not os.path.exists(asr_source_file):
                    continue

                asr_eval_file = os.path.join("data", "results", "evaluation", "asr",
                                             f"asr_eval_{src_lang}_{asr_folder_name}.json")

                if not os.path.exists(asr_eval_file):
                    print(
                        f"❌ Skipping {asr_short_name} segment {speaker_group}_{condition_code}: ASR evaluation JSON not found.")
                    continue

                print(f"  --- Segment: {speaker_group.upper()}_{condition_code.upper()} ---")

                current_mt_models = {
                    k: v for k, v in MT_MODELS.items()
                    if f"{src_lang}_{tgt_lang}" in k or k.startswith("nllb")
                }

                for mt_short_name, mt_folder_name in current_mt_models.items():
                    print(f"--- MT Model: {mt_short_name.upper()} ---")

                    model_part = mt_short_name.split("_")[0]

                    pred_file_name = f"{asr_output_base_name}_{model_part}_{src_lang}2{tgt_lang}_results.txt"

                    pred_file = os.path.join(
                        "data", "results", "mt",
                        mt_folder_name,
                        pred_file_name
                    )

                    if not os.path.exists(pred_file):
                        print(f" ❌ Skipping: MT Prediction file not found: {pred_file}")
                        continue

                    if not os.path.exists(pred_file):
                        pred_file = os.path.join(
                            "data", "results", "mt",
                            pred_file_name
                        )

                    command = [
                        sys.executable,
                        mt_script_path,
                        "--model", f"{mt_short_name}",
                        "--src_lang", src_lang,
                        "--tgt_lang", tgt_lang,
                        "--src_file", asr_source_file,
                        "--ref_file", ref_file,
                        "--pred_file", pred_file,
                        "--asr_wer_status", asr_eval_file
                    ]

                    try:
                        subprocess.run(
                            command,
                            check=True,
                            text=True,
                            cwd=os.getcwd(),
                            stdout=sys.stdout,
                            stderr=sys.stderr
                        )
                        print(f"  ✅ MT evaluation complete: {mt_short_name} / Source: {asr_short_name}")
                    except subprocess.CalledProcessError as e:
                        print(f"  ❌ ERROR evaluating MT task: {mt_short_name} / Source: {asr_short_name}!")
                        print(f"  Command: {' '.join(e.cmd)}")
                        stderr_content = e.stderr.strip() if e.stderr else "(No stderr output)"
                        print(f"❌ Standard Error (Stderr): {stderr_content}")
                        print("-------------------------------------------------")
                    except FileNotFoundError:
                        print(f"❌ FATAL ERROR: evaluate_mt.py script file not found.")
                        raise

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

    for f in asr_files:
        try:
            df = pd.read_json(f)
            df['Task'] = 'ASR'
            all_data.append(df)
        except Exception as e:
            print(f"❌ ERROR reading ASR file {f}: {e}")

    for f in mt_files:
        try:
            df = pd.read_json(f)
            df['Task'] = 'MT'
            all_data.append(df)
        except Exception as e:
            print(f"❌ ERROR reading MT file {f}: {e}")

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)

        cols = ['Task', 'Model', 'Language', 'Audio_File_Name', 'Line_Index',
                'Speaker_Group', 'Speaker_Type', 'Speaker_Gender',
                'Condition_Code', 'Noise_Level', 'Noise_Type', 'Voice_Volume',
                'Source_Lang', 'Target_Lang', 'ASR_WER_Status', 'High_WER_Percentage',
                'WER', 'WIL', 'METEOR', 'TER', 'COMET']

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

    # 1. ASR Evaluation (generates WER data per row)
    normalize_all_references()
#    for lang in SHORT_LANG_CODES:
 #       run_asr_evaluation(lang)

    # 2. MT Evaluation with Human Source
 #   for src_lang in SHORT_LANG_CODES:
  #      for tgt_lang in SHORT_LANG_CODES:
   #         if src_lang != tgt_lang:
    #            run_mt_evaluation_on_human_source(src_lang, tgt_lang)

    # 3. MT Evaluation with ASR Output Source (uses ASR evaluation JSON for quality flagging)
    for src_lang in SHORT_LANG_CODES:
        for tgt_lang in SHORT_LANG_CODES:
            if src_lang != tgt_lang:
                run_mt_evaluation_on_asr_source(src_lang, tgt_lang)

    compile_results()

    print("=" * 70)
    print(" ALL EVALUATIONS COMPLETED ")
    print("=" * 70)

if __name__ == "__main__":
    main()