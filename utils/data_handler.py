import os
import re
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
    """

    lang_code_short = lang_code_full[:2].lower()
    model_safe_name = model_id.replace("/", "_").replace("-", "_").replace(".", "_")

    output_dir = os.path.join("data", "results", "asr", model_safe_name, lang_code_short)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    print(f"[ASR Handler] Saving {len(transcriptions)} transcription(s) to file {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in transcriptions:
                f.write(text.strip() + "\n")
            print("Saving complete.")
    except Exception as e:
        print(f"❌ ERROR saving ASR results to file {output_path}: {e}")

def save_asr_single_result(
        transcription: str,
        model_id: str,
        output_filename_with_metadata: str
):
    """
    Saves a single ASR transcription to a unique file, ensuring metadata is preserved
    in the filename (e.g., fi_nat_m_c1_transcription.txt).

    :param transcription: The single resulting text.
    :param model_id: Model ID or name
    :param output_filename_with_metadata: The full desired filename (e.g., 'fi_nat_m_c1_transcription.txt').
    """

    model_safe_name = model_id.replace("/", "_").replace("-", "_").replace(".", "_")

    output_dir = os.path.join("data", "results", "asr", model_safe_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename_with_metadata)

    print(f"[ASR Handler] 💾 Saving single transcription to file: {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcription.strip() + "\n")
            print("Saving complete.")
    except Exception as e:
        print(f"❌ ERROR saving ASR results to file {output_path}: {e}")

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

def save_results_mt(results: List[str], output_path: str):
    """
    Saves translation results to file for evaluation.py.

    :param results: List of translation results
    :param output_path: Path for result file
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    print(f"[MT Handler] Saving translation results to file: {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in results:
                f.write(text + "\n")
            print("✅ Saving complete.")
    except Exception as e:
        print(f"❌ ERROR saving results to {output_path}: {e}")

def write_data(data: List[str], filepath: str):
    """Writes data (list of rows) to file. """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for line in data:
                f.write(line + '\n')
        return True
    except Exception as e:
        print(f"❌ ERROR writing data to {filepath}: {e}")
        return False

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