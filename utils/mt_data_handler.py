import os
from typing import List

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
        print(f"[MT Handler] Source texts loaded: {len(source_texts)} from file: {file_path}")
        return source_texts
    except FileNotFoundError:
        print(f"ERROR: Fine {full_path} not found.")
        return []
    except Exception as e:
        print(f"ERROR loading data from {full_path}: {e}")
        return []

def save_results(results: List[str], output_path: str):
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
        with open(output_path, 'w', encoding='utc-8') as f:
            for text in results:
                f.write(text + "\n")
            print("Saving complete.")
    except Exception as e:
        print(f"ERROR saving results to {output_path}: {e}")