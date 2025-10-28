from typing import Literal, Dict, List

SHORT_LANG_CODES: List[str] = ["fi", "en", "fr"]

ASR_LANG_CODES_FULL: List[str] = ["finnish", "english", "french"]

NLLB_LANG_MAP = {
    "fi": "fin_Latn",
    "en": "eng_Latn",
    "fr": "fra_Latn"
}

BLOOM_LANG_MAP = {
    "fi": "Finnish",
    "en": "English",
    "fr": "French"
}

MT_ALLOWED_LANGUAGES = Literal["fi", "en", "fr"]

ASR_ALLOWED_LANGUAGES = Literal["finnish", "english", "french"]