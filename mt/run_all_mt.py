import argparse
import subprocess
import os
import sys
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.constants import SHORT_LANG_CODES

MODEL_MAP = {
    "nllb": "nllb.py",
    "opus": "opus_models.py",
}
def run_translation_for_file(script_path, src_lang, src_file, tgt_lang, model_name):
    """
    Runs all translations for all target languages from one source file.
    """

    print(f"\n🚀 Running translation: {src_lang} -> {tgt_lang}...")

    command = [
        sys.executable,
        script_path,
        "--src_lang", src_lang,
        "--tgt_lang", tgt_lang,
        "--src_file", src_file
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

        print(f"✅ Translation ({src_lang} -> {tgt_lang}) complete.")

    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR translating {src_lang} -> {tgt_lang}!")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Model Standard Error (Stderr): {e.stderr.strip()}")

    except FileNotFoundError:
        print(f"❌ FATAL ERROR: Python script file not found.")
        raise


def get_source_language(filename: str) -> str | None:
    """Parses source language from file name prefix ('fi_transkriptio.txt' -> 'fi')."""
    prefix = filename[:2].lower()

    if prefix in SHORT_LANG_CODES:
        return prefix
    return None

def main():
    """
    Runs all translations for all .txt files in the defined source directory.
    """
    parser = argparse.ArgumentParser(
        description="Run all translation tasks for defined model and source language.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", type=str, required=True, choices=MODEL_MAP.keys(),
                        help="Model name: nllb or opus.")
    parser.add_argument("--src_dir", type=str, required=True,
                        help="Path to source directory (e.g., data/fr).")
    args = parser.parse_args()

    src_dir = args.src_dir
    model_name = args.model

    model_script = MODEL_MAP.get(model_name)
    script_path = os.path.join(os.path.dirname(__file__), model_script)

    if not os.path.exists(script_path):
        print(f"❌ FATAL ERROR: Model script not found: {script_path}")
        return

    source_files = sorted(glob.glob(os.path.join(src_dir, "*.txt")))

    if not source_files:
        print(f"⚠️ No .txt files found in source directory: {src_dir}")
        return

    print("=" * 50)
    print(f"🔥 Starting batch run for model: {model_name.upper()} ({model_script})")
    print(f"Source directory: {src_dir}")
    print(f"Files to process: {len(source_files)}")
    print("=" * 50)

    for src_file in source_files:
        filename = os.path.splitext(os.path.basename(src_file))[0]
        src_lang = get_source_language(filename)

        if not src_lang:
            print(f"⚠️ SKIPPING: Could not determine source language from filename '{filename}'. Must start with fi_, en_ or fr_.")
            continue

        tgt_langs = [lang for lang in SHORT_LANG_CODES if lang != src_lang]

        if not tgt_langs:
            print(f"❌ ERROR: Target language not found for '{src_lang}'. Check SHORT_LANG_CODES.")
            return

        print("\n" + "~" * 40)
        print(f"Processing file: **{os.path.basename(src_file)}**")
        print(f"Target languages: {', '.join(tgt_langs)}")
        print("~" * 40)

        for tgt_lang in tgt_langs:
            run_translation_for_file(script_path, src_lang, src_file, tgt_lang, model_name)

    print("\n" + "=" * 50)
    print("✨ Batch run complete. See results in 'results/mt/'.")
    print("=" * 50)

if __name__ == "__main__":
    main()