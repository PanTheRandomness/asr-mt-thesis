from typing import Literal, List, Tuple, Dict

SHORT_LANG_CODES: List[str] = ["fi", "en", "fr"]

ASR_LANG_CODES_FULL: List[str] = ["finnish", "english", "french"]

NLLB_LANG_MAP = {
    "fi": "fin_Latn",
    "en": "eng_Latn",
    "fr": "fra_Latn"
}

WAV2VEC2_MODEL_MAP = {
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "en": "facebook/wav2vec2-large-960h",
    "fr": "facebook/wav2vec2-large-xlsr-53-french"
}

OPUS_MODEL_MAP: dict[Tuple[str, str], str] = {
    ("fi", "en"): "Helsinki-NLP/opus-mt-fi-en",
    ("en", "fi"): "Helsinki-NLP/opus-mt-en-fi",
    ("fi", "fr"): "Helsinki-NLP/opus-mt-fi-fr",
    ("fr", "fi"): "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-fiu",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en"
}

ASR_MODELS: Dict[str, str] = {
    "whisper_large_v2": "openai_whisper_large_v2",
    "wav2vec2_xlsr_fi": "jonatasgrosman_wav2vec2_large_xlsr_53_finnish",
    "wav2vec2_xlsr_fr": "facebook_wav2vec2_large_xlsr_53_french",
    "wav2vec2_xlsr_en": "facebook_wav2vec2_large_960h",
    "parakeet": "nvidia_parakeet_tdt_0_6b_v3",
}

MT_MODELS: Dict[str, str] = {
    "nllb": "facebook_nllb-200-distilled-600M",
    "opus_mt_fi_en": "opus-mt-fi-en",
    "opus_mt_en_fi": "opus-mt-en-fi",
    "opus_mt_fi_fr": "opus-mt-fi-fr",
    "opus_mt_fr_fi": "opus-mt-tc-bible-big-deu_eng_fra_por_spa-fiu",
    "opus_mt_en_fr": "opus-mt-en-fr",
    "opus_mt_fr_en": "opus-mt-fr-en"
}

MT_ALLOWED_LANGUAGES = Literal["fi", "en", "fr"]

ASR_ALLOWED_LANGUAGES = Literal["finnish", "english", "french"]

CONDITIONS_MAP = {
    "c1": {
        "background_noise_level": "low/none",
        "background_noise_type": None,
        "voice_volume": "normal"
    },
    "c2": {
        "background_noise_level": "low/none",
        "background_noise_type": None,
        "voice_volume": "low"
    },
    "c3": {
        "background_noise_level": "medium",
        "background_noise_type": "human",
        "voice_volume": "normal"
    },
    "c4": {
        "background_noise_level": "medium",
        "background_noise_type": "traffic",
        "voice_volume": "normal"
    },
    "c5": {
        "background_noise_level": "high",
        "background_noise_type": "human",
        "voice_volume": "normal"
    },
    "c6": {
        "background_noise_level": "high",
        "background_noise_type": "traffic",
        "voice_volume": "normal"
    }
}