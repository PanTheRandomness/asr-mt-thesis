import os
from typing import List

def save_asr_results(
        transcriptions: List[str],
        model_id: str,
        lang_code_full: str,
        output_filename: str = "predictions.txt"
):
    """
    Saves ASR-transcriptions to the output file for evaluation.

    :param transcriptions: List of all transcriptions (predictions).
    :param model_id: Model ID or name
    :param lang_code_full: Long language name (e.g., "finnish")
    :param output_filename: Name of output file.
    :return:
    """

    lang_code_short = lang_code_full[:2].lower()
    model_safe_name = model_id.replace("/", "_").replace("-", "_").replace(".", "_")

    output_dir = os.path.join("data", "results", "asr", model_safe_name, lang_code_short)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    print(f"[ASR Handler] Saving {len(transcriptions)} transcription(s) to file {output_path}")
    try:
        with open(output_path, 'w', encoding='utc-8') as f:
            for text in transcriptions:
                f.write(text.strip() + "\n")
            print("Saving complete.")
    except Exception as e:
        print(f"ERROR saving ASR results to file {output_path}: {e}")

