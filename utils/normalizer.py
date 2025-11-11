import re

def asr_normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[.,:;!?"\'\\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text