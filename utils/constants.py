from typing import Literal, List, Tuple

SHORT_LANG_CODES: List[str] = ["fi", "en", "fr"]

ASR_LANG_CODES_FULL: List[str] = ["finnish", "english", "french"]

NLLB_LANG_MAP = {
    "fi": "fin_Latn",
    "en": "eng_Latn",
    "fr": "fra_Latn"
}

OPUS_MODEL_MAP: dict[Tuple[str, str], str] = {
    ("fi", "en"): "Helsinki-NLP/opus-mt-fi-en",
    ("en", "fi"): "Helsinki-NLP/opus-mt-en-fi",
    ("fi", "fr"): "Helsinki-NLP/opus-mt-fi-fr",
    ("fr", "fi"): "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-fiu",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en"
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