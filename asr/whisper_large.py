import librosa
import os
from ..utils.model_loader import load_whisper_asr_model
from ..utils.constants import ASR_ALLOWED_LANGUAGES, ASR_LANG_CODES_FULL

WHISPER_MODEL_NAME = "openai/whisper-large-v2"
WHISPER_MODEL, WHISPER_PROCESSOR, DEVICE = load_whisper_asr_model(WHISPER_MODEL_NAME)

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

    if target_language not in ASR_LANG_CODES_FULL:
        print(f"ERROR: Language '{target_language}' not supported by this script")
        return ""

    try:
        audio_input, _ = librosa.load(audio_path, sr=16000)
    except Exception as e:
        print(f"ERROR in audio processing: {e}")
        return ""

    input_features = WHISPER_PROCESSOR(
        audio_input,
        sampling_rate=16000,
        return_tensors="pt"
    ).input__features

    if DEVICE.startswith("cuda"):
        input_features = input_features.to(DEVICE)

    decoder_ids = WHISPER_PROCESSOR.get_decoder_prompt_ids(
        language=target_language,
        task=asr_task
    )

    predicted_ids = WHISPER_MODEL.generate(
        inputs=input_features,
        forced_decoder_ids=decoder_ids,
        max_length=448
    )

    transcription = WHISPER_PROCESSOR.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription
