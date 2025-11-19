import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob

from transformers import pipeline

from utils.model_loader import load_whisper_asr_model
from utils.constants import SHORT_LANG_CODES, ASR_LANG_CODES_FULL, ASR_ALLOWED_LANGUAGES
from utils.data_handler import save_asr_single_result
from utils.sentence_splitter import SentenceSplitter

WHISPER_MODEL_NAME = "openai/whisper-large-v2"
MODEL_SHORT_NAME = WHISPER_MODEL_NAME.split("/")[-1]
WHISPER_MODEL, WHISPER_PROCESSOR, DEVICE = load_whisper_asr_model(WHISPER_MODEL_NAME)

WHISPER_PIPE = None
if WHISPER_MODEL is not None and WHISPER_PROCESSOR is not None:
    print(f"[ASR] Creating ASR pipeline on device: {DEVICE}.")
    try:
        WHISPER_PIPE = pipeline(
            "automatic-speech-recognition",
            model=WHISPER_MODEL,
            tokenizer=WHISPER_PROCESSOR.tokenizer,
            feature_extractor=WHISPER_PROCESSOR.feature_extractor
        )
    except Exception as e:
        print(f"❌ ERROR creating pipeline: {e}")
        exit()

if WHISPER_MODEL is None:
    print("❌ Model loading failed. Terminating process.")
    exit()
else:
    print(f"✅ {WHISPER_MODEL_NAME} has been loaded and is in use in {DEVICE}")

def transcribe_and_save_single_file(audio_path: str, short_lang: str):
    filename = os.path.basename(audio_path)
    metadata_suffix = filename.replace("-", "_").split('.')[0]
    output_filename = f"{metadata_suffix}_transcription.txt"

    output_check_path = os.path.join("data", "results", "asr", MODEL_SHORT_NAME, output_filename)
    if os.path.exists(output_check_path):
        print(f"Skipping {filename}: results already exists at {output_check_path}.")

    transcription = transcribe_audio_whisper(
        audio_path=audio_path,
        target_language=short_lang
    )

    if transcription:
        final_output = SentenceSplitter.split_and_clean(transcription)

        save_asr_single_result(
            transcription=final_output,
            model_id=WHISPER_MODEL_NAME,
            output_filename_with_metadata=output_filename
        )
    else:
        print(f"❌ Failed to transcribe {filename}.")

def transcribe_audio_whisper(
        audio_path: str,
        target_language: ASR_ALLOWED_LANGUAGES,
        asr_task: str = "transcribe"
) -> str:
    """
    Creates a Transcription with Whisper Large v2.

    :param audio_path: Path to audio file
    :param target_language: Target language (used as metadata)
    :param asr_task: "transcribe" or "translate"
    :return: transcription
    """

    if not os.path.exists(audio_path):
        print(f"❌ ERROR: File '{audio_path}' not found.")
        return ""

    try:
        result = WHISPER_PIPE(
            audio_path,
            chunk_length_s=30,
            stride_length_s=(4, 2),
            generate_kwargs={
                "language": target_language,
                "task": asr_task
            },
            return_timestamps=True,
            ignore_warning=True
        )
        return result["text"]
    except Exception as e:
        print(f"❌ ERROR in audio processing: {e}")
        return ""

def run_whisper_transcription_on_datasets(langs: list[str] = SHORT_LANG_CODES):
    """
    Main function fort Whisper ASR task.
    Iterates through all 24 audio files per language (4 speakers x 6 conditions) and saves the transcriptions.
    """

    lang_map = dict(zip(SHORT_LANG_CODES, ASR_LANG_CODES_FULL))

    print(f"Starting ASR run for {MODEL_SHORT_NAME}...")

    for short_lang in langs:
        full_lang = lang_map.get(short_lang)
        if not full_lang:
            print(f"❌ ERROR: Language core {short_lang} not recognised.")
            continue

        data_dir = os.path.join("data", short_lang, f"{short_lang}-*.wav")
        audio_files = glob.glob(data_dir)

        if not audio_files:
            print(f"⚠️ No WAV files found for language {short_lang} in {data_dir}.")
            continue

        print(f"\n--- Starting Whisper transcription for {full_lang} ({len(audio_files)} files) ---")

        for i, audio_path in enumerate(audio_files):
            print(f"[{i + 1}/{len(audio_files)}] Transcribing: {audio_path}...")
            transcribe_and_save_single_file(audio_path, short_lang)

if __name__ == "__main__":
    if WHISPER_MODEL is None:
        exit()

    # python asr/whisper_large.py path_to_audio_file.wav
    if len(sys.argv) > 1 and sys.argv[1].endswith(".wav"):
        single_file_path = sys.argv[1]
        print(f"\n--- Starting single file transcription: {single_file_path} ---")

        path_parts = single_file_path.split(os.sep)
        try:
            lang_index = path_parts.index("data") + 1
            if lang_index < len(path_parts) and path_parts[lang_index] in SHORT_LANG_CODES:
                lang_code = path_parts[lang_index]
            else:
                raise ValueError()
        except (ValueError, IndexError):
            print(f"❌ ERROR: Could not determine language codes (fi/en/fr) from file path. Terminating.")
            exit(1)

        if os.path.exists(single_file_path):
            transcribe_and_save_single_file(single_file_path, lang_code)
    else:
        # Run for all languages in SHORT_LANG_CODES (default: fi, en, fr)
        # NOTE: Run from project root directory!
        # python asr/whisper_large.py
        run_whisper_transcription_on_datasets()