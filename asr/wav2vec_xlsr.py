import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import librosa
import torch

from utils.model_loader import load_wav2vec2_asr_model
from utils.constants import SHORT_LANG_CODES, ASR_LANG_CODES_FULL, WAV2VEC2_MODEL_MAP
from utils.data_handler import save_asr_single_result
from utils.sentence_splitter import SentenceSplitter

WAV2VEC2_MODEL = None
WAV2VEC2_PROCESSOR = None
DEVICE = None
WAV2VEC2_MODEL_NAME = None

def transcribe_and_save_single_file(audio_path: str, model_id: str):
    model_short_name = model_id.split("/")[-1]
    filename = os.path.basename(audio_path)
    output_filename = f"{os.path.splitext(filename)[0]}_transcription.txt"

    output_check_path = os.path.join("data", "results", "asr", model_short_name, output_filename)
    if os.path.exists(output_check_path):
        print(f"Skipping {filename}: results already exists at {output_check_path}.")

    transcription = transcribe_audio_wav2vec2(
        audio_path=audio_path
    )

    if transcription:
        final_output = SentenceSplitter.split_and_clean(transcription)

        save_asr_single_result(
            transcription=final_output,
            model_id=WAV2VEC2_MODEL_NAME,
            output_filename_with_metadata=output_filename
        )
    else:
        print(f"❌ Failed to transcribe {filename}.")

def transcribe_audio_wav2vec2(
        audio_path: str
) -> str:
    """
    Creates a Transcription with Wav2Vec 2.0 XLSR-53 (CTC).

    :param audio_path: Path to audio file (16 kHz, mono)
    :return: Transcription
    """

    global WAV2VEC2_MODEL, WAV2VEC2_PROCESSOR, DEVICE

    if WAV2VEC2_MODEL is None:
        print("❌ ERROR: Model not loaded. Cannot transcribe.")
        return ""

    if not os.path.exists(audio_path):
        print(f"❌ ERROR: File '{audio_path}' not found.")
        return ""

    try:
        audio_input, _ = librosa.load(audio_path, sr=16000)
    except Exception as e:
        print(f"❌ ERROR in audio processing: {e}")
        return ""

    input_values = WAV2VEC2_PROCESSOR(
        audio_input,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_values

    if DEVICE.startswith("cuda"):
        input_values = input_values.to(DEVICE)
        input_values = input_values.to(WAV2VEC2_MODEL.dtype)

    try:
        with torch.no_grad():
            logits = WAV2VEC2_MODEL(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)

        transcription = WAV2VEC2_PROCESSOR.batch_decode(predicted_ids)[0]
        return transcription.lower()

    except Exception as e:
        print(f"❌ ERROR during Wav2Vec2 forward pass: {e}")
        return ""

def run_wav2vec_transcription_on_datasets(langs: list[str] = SHORT_LANG_CODES):
    """
    Main function fort Wav2Vec ASR task.
    """

    MODEL_SHORT_NAME = WAV2VEC2_MODEL_NAME.split("/")[-1]

    lang_map = dict(zip(SHORT_LANG_CODES, ASR_LANG_CODES_FULL))

    for short_lang in langs:
        full_lang = lang_map.get(short_lang)
        if not full_lang:
            print(f"❌ ERROR: Language code {short_lang} not recognised.")
            continue

        data_dir = os.path.join("data", short_lang, f"{short_lang}-*.wav")
        audio_files = glob.glob(data_dir)

        if not audio_files:
            print(f"⚠️ No WAV files found for language {full_lang} in {data_dir}.")
            continue

        print(f"\n--- Starting Wav2Vec2 transcription for {full_lang} ({len(audio_files)} files) ---")

        for i, audio_path in enumerate(audio_files):
            filename = os.path.basename(audio_path)
            metadata_suffix = filename.replace("-", "_").split('.')[0]
            output_filename = f"{metadata_suffix}_transcription.txt"

            output_check_path = os.path.join("data", "results", "asr", MODEL_SHORT_NAME, output_filename)
            if os.path.exists(output_check_path):
                print(f"Skipping {filename}: results already exists at {output_check_path}.")
                continue

            print(f"[{i + 1}/{len(audio_files)}] Transcribing: {audio_path}...")

            transcription = transcribe_audio_wav2vec2(
                audio_path=audio_path
            )

            if transcription:
                final_output = SentenceSplitter.split_and_clean(transcription)

                save_asr_single_result(
                    transcription=final_output,
                    model_id=WAV2VEC2_MODEL_NAME,
                    output_filename_with_metadata=output_filename
                )
            else:
                print(f"❌ Failed to transcribe {filename}.")

if __name__ == "__main__":

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
            print("❌ ERROR: Could not determine language code (fi/en/fr) from file path. Terminating.")
            exit(1)

        WAV2VEC2_MODEL_NAME = WAV2VEC2_MODEL_MAP[lang_code]
        WAV2VEC2_MODEL, WAV2VEC2_PROCESSOR, DEVICE = load_wav2vec2_asr_model(WAV2VEC2_MODEL_NAME)

        if WAV2VEC2_MODEL is None:
            print(f"❌ Model loading failed for {lang_code}. Terminating.")
            exit()

        if os.path.exists(single_file_path):
            transcribe_and_save_single_file(single_file_path, WAV2VEC2_MODEL_NAME)
        else:
            print(f"❌ ERROR: Specified file not found: {single_file_path}")

    else:
        for lang_code, model_name in WAV2VEC2_MODEL_MAP.items():
            print(f"\n" + 25 * "=")
            print(f"SWITCHING LANGUAGE: {lang_code.upper()} using model: {model_name}.")
            print(f"\n" + 25 * "=")

            WAV2VEC2_MODEL_NAME = model_name

            WAV2VEC2_MODEL, WAV2VEC2_PROCESSOR, DEVICE = load_wav2vec2_asr_model(WAV2VEC2_MODEL_NAME)

            if WAV2VEC2_MODEL is None:
                print(f"❌ Model loading failed for {lang_code}. Skipping this language.")
                continue

            print(f"✅ {WAV2VEC2_MODEL_NAME} has been loaded and is in use in {DEVICE}.")

            run_wav2vec_transcription_on_datasets(langs=[lang_code])