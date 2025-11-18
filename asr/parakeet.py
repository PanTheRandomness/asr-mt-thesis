import sys
import os
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import librosa

from utils.model_loader import load_nemo_asr_model
from utils.constants import ASR_LANG_CODES_FULL
from utils.asr_data_handler import save_asr_single_result
from utils.sentence_splitter import SentenceSplitter

PARAKEET_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
MODEL_SHORT_NAME = PARAKEET_MODEL_NAME.split("/")[-1]
PARAKEET_MODEL, _, DEVICE = load_nemo_asr_model(PARAKEET_MODEL_NAME)

if PARAKEET_MODEL is None:
    print("❌ Model loading failed. Terminating process.")
    exit()
else:
    print(f"✅ NeMo {PARAKEET_MODEL_NAME} has been loaded and is in use in {DEVICE}")

def transcribe_and_save_single_file(audio_path: str, model_id: str):
    """ Transcribes and saves the result for a single audio file, including checkpointing. """

    model_short_name = model_id.split("/")[-1]
    filename = os.path.basename(audio_path)
    base_file_name = os.path.splitext(filename)[0]
    output_filename = f"{base_file_name}_transcription.txt"

    output_check_path = os.path.join("data", "results", "asr", model_short_name, output_filename)
    if os.path.exists(output_check_path):
        print(f"Skipping {filename}: results already exists at {output_check_path}.")
        return

    print(f"Transcribing: {audio_path}...")
    transcription = transcribe_audio_parakeet(audio_path)

    if transcription and "ERROR" not in transcription:
        final_output = SentenceSplitter.split_and_clean(transcription)

        save_asr_single_result(
            transcription=final_output,
            model_id=model_id,
            output_filename_with_metadata=output_filename
        )
        print(f"✅ Transcription saved for {filename}.")
    else:
        print(f"❌ Failed to transcribe {audio_path}.")

def transcribe_audio_parakeet(audio_path: str) -> str:
    """
    Creates a Transcription with Parakeet-tdt-0.6b-v3 (NeMo version).
    Parakeet recognises the language automatically from 25 european languages.

    :param audio_path: Path to audio file (recommended: WAV, 16 kHz, mono)
    :return: Transcription
    """

    if PARAKEET_MODEL is None:
        return "❌ ERROR: No model loaded."

    if not os.path.exists(audio_path):
        print(f"❌ ERROR: File '{audio_path}' not found.")
        return ""

    TARGET_SR = 16000

    try:
        print(f"[ASR FIX] Loading audio with librosa (16kHz, mono): {os.path.basename(audio_path)}")
        audio_data, _ = librosa.load(
            audio_path,
            sr=TARGET_SR,
            mono=True
        )
        signal = torch.tensor(audio_data,dtype=torch.float32).to(DEVICE)
        signal = signal.unsqueeze(0)
        audio_len = torch.tensor([signal.shape[-1]], dtype=torch.long).to(DEVICE)

        with torch.no_grad():
           processed_signal, processed_signal_length = PARAKEET_MODEL.preprocessor(
               input_signal=signal,
               length=audio_len
           )

           encoder_output, encoder_len = PARAKEET_MODEL.forward(
               processed_signal=processed_signal,
               processed_signal_length=processed_signal_length
           )

           hypotheses = PARAKEET_MODEL.decoding.rnnt_decoder_predictions_tensor(
               encoder_output, encoder_len
           )

        if hypotheses and hasattr(hypotheses[0], 'text'):
            raw_transcription = hypotheses[0].text

            final_transcription = SentenceSplitter.split_and_clean(raw_transcription)
            return final_transcription
        else:
            return "❌ Transcription failed (empty result)."

    except Exception as e:
        print(f"❌ ERROR running NeMo model on file {audio_path}: {e}")
        return ""

def run_parakeet_transcription_on_dataset():
    """
    Processes all WAV files for all defined languages in the data/ directory and saves the combined results using
    the ASR data handler.
    """
    model_id = PARAKEET_MODEL_NAME

    for full_lang in ASR_LANG_CODES_FULL:
        short_lang_code = full_lang[:2].lower()

        data_dir = os.path.join("data", short_lang_code)
        audio_files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)

        if not audio_files:
            print(f"⚠️ No WAV files found for language {full_lang} in {data_dir}.")
            continue

        print(f"\n--- Starting Parakeet transcription for {full_lang} ({len(audio_files)} files) ---")

        for i, audio_path in enumerate(audio_files):
            transcribe_and_save_single_file(audio_path, model_id)

        print(f"--- {full_lang} transcription complete ---")

if __name__ == "__main__":
    if PARAKEET_MODEL is None:
        exit()

    if len(sys.argv) > 1 and sys.argv[1].endswith(".wav"):
        single_file_path = sys.argv[1]
        print(f"\n--- Starting single file transcription: {single_file_path} ---")
        if os.path.exists(single_file_path):
            transcribe_and_save_single_file(single_file_path, PARAKEET_MODEL_NAME)
        else:
            print(f"❌ ERROR: Specified file not found: {single_file_path}")
    else:
        run_parakeet_transcription_on_dataset()