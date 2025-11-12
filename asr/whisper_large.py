import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import re

from transformers import pipeline

from utils.model_loader import load_whisper_asr_model
from utils.constants import SHORT_LANG_CODES, ASR_LANG_CODES_FULL, ASR_ALLOWED_LANGUAGES
from utils.asr_data_handler import save_asr_single_result

WHISPER_MODEL_NAME = "openai/whisper-large-v2"
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
        print(f"ERROR creating pipeline: {e}")
        exit()

if WHISPER_MODEL is None:
    print("Model loading failed. Terminating process.")
    exit()
else:
    print(f"✅ {WHISPER_MODEL_NAME} has been loaded and is in use in {DEVICE}")

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
        print(f"ERROR: File '{audio_path}' not found.")
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
            return_timestamps=True
        )
        return result["text"]
    except Exception as e:
        print(f"ERROR in audio processing: {e}")
        return ""

def main(langs: list[str] = SHORT_LANG_CODES):
    """
    Main function fort Whisper ASR task.
    Iterates through all 24 audio files per language (4 speakers x 6 conditions) and saves the transcriptions.
    """

    MODEL_SHORT_NAME = WHISPER_MODEL_NAME.split("/")[-1]

    lang_map = dict(zip(SHORT_LANG_CODES, ASR_LANG_CODES_FULL))

    print(f"Starting ASR run for {MODEL_SHORT_NAME}...")

    for short_lang in langs:
        full_lang = lang_map.get(short_lang)
        if not full_lang:
            print(f"ERROR: Language core {short_lang} not recognised.")
            continue

        audio_pattern = os.path.join("data", short_lang, f"{short_lang}-*.wav")
        audio_files = glob.glob(audio_pattern)

        if not audio_files:
            print(f"No audio files found for language {short_lang} in {audio_pattern}.")
            continue

        print(f"Found {len(audio_files)} files for {full_lang}. Processing...")

        for audio_path in audio_files:
            filename = os.path.basename(audio_path)
            metadata_suffix = filename.replace("-", "_").split('.')[0]
            output_filename = f"{metadata_suffix}_transcription.txt"

            output_check_path = os.path.join("data", "results", "asr", MODEL_SHORT_NAME, output_filename)
            if os.path.exists(output_check_path):
                print(f"Skipping {filename}: results already exists at {output_check_path}.")
                continue

            print(f"--> Transcribing {filename}...")

            transcription = transcribe_audio_whisper(
                audio_path=audio_path,
                target_language=short_lang
            )

            if transcription:
                sentences = re.split(r'([.!?])\s', transcription.strip())

                final_sentences = []
                for i in range(0, len(sentences), 2):
                    sentence_segment = sentences[i].strip()
                    if sentence_segment:
                        if i + 1 < len(sentences):
                            sentence_segment += sentences[i+1]
                        final_sentences.append(sentence_segment)

                final_output = '\n'.join(final_sentences)

                if not final_sentences:
                    # Save raw transcription if segmentation fails
                    final_output = transcription.strip()

                save_asr_single_result(
                    transcription=final_output,
                    model_id=WHISPER_MODEL_NAME,
                    output_filename_with_metadata=output_filename
                )
            else:
                print(f"❌ Failed to transcribe {filename}.")


if __name__ == "__main__":
    # Run for all languages in SHORT_LANG_CODES (default: fi, en, fr)
    # NOTE: Run from project root directory!
    # python asr/whisper_large.py
    main()