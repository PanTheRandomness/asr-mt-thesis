import argparse
import subprocess
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.constants import SHORT_LANG_CODES

MODEL_MAP = {
    "nllb": "nllb.py",
    "opus": "opus_models.py",
}


def main():
    """
    Runs all translations to all target languages from one source file.
    """
    parser = argparse.ArgumentParser(
        description="Run all translation tasks for defined model and source language.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", type=str, required=True, choices=MODEL_MAP.keys(),
                        help="Model name: nllb or opus.")
    parser.add_argument("--src_lang", type=str, required=True, choices=SHORT_LANG_CODES,
                        help="Source language (fi, en, fr).")
    parser.add_argument("--src_file", type=str, required=True,
                        help="Path to source file (e.g., data/fr/test_set.txt).")
    args = parser.parse_args()

    src_lang = args.src_lang
    src_file = args.src_file
    model_name = args.model

    tgt_langs = [lang for lang in SHORT_LANG_CODES if lang != src_lang]

    if not tgt_langs:
        print(f"❌ ERROR: Target language not found for '{src_lang}'. Check SHORT_LANG_CODES.")
        return

    model_script = MODEL_MAP.get(model_name)
    script_path = os.path.join(os.path.dirname(__file__), model_script)

    print("=" * 50)
    print(f"🔥 Starting batch run for model: {model_name.upper()} ({model_script})")
    print(f"Source language: {src_lang} | Source file: {src_file}")
    print(f"Target languages: {', '.join(tgt_langs)}")
    print("=" * 50)

    for tgt_lang in tgt_langs:
        print(f"\n🚀 Running translation: {src_lang} -> {tgt_lang}...")

        command = [
            sys.executable,
            script_path,
            "--src_lang", src_lang,
            "--tgt_lang", tgt_lang,
            "--src_file", src_file
        ]

        try:
            subprocess.run(command, check=True, capture_output=True, text=True, cwd=os.getcwd())

            print(f"✅ Translation ({src_lang} -> {tgt_lang}) complete.")

        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR translating {src_lang} -> {tgt_lang}!")
            print(f"Command: {' '.join(e.cmd)}")
            print(f"Model Standard Error (Stderr): {e.stderr.strip()}")

        except FileNotFoundError:
            print(f"❌ FATAL ERROR: Python script file not found.")
            break

    print("\n" + "=" * 50)
    print("✨ Batch run complete. See results in 'results/mt/'.")
    print("=" * 50)


if __name__ == "__main__":
    main()