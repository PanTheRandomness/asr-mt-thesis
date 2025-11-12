import re

# NOTE: For translation it would maybe make more sense to have the sentences in one line.
class SentenceSplitter:
    ABBREVIATIONS = r'\bM|\bMme|\bMlle|\bMdlle|\bMr|\bMrs|\bDr|\bMs|\betc|\besim|\bks|\bjne|\byms'

    ABBREVIATION_MATCH_PATTERN = re.compile(
        r'(' + ABBREVIATIONS + r')\.\s*',
        flags=re.IGNORECASE
    )

    PLACEHOLDER = "[#ASR_SPLIT_SKIP#]"

    @staticmethod
    def split_and_clean(text: str) -> str:
        if not text:
            return ""

        temp_text = text.strip()

        def replace_abbreviations(match):
            return match.group(1) + SentenceSplitter.PLACEHOLDER

        temp_text = SentenceSplitter.ABBREVIATION_MATCH_PATTERN.sub(
            replace_abbreviations,
            temp_text
        )

        separated_text = re.sub(r'\.\s*', r'.\n', temp_text)
        final_text = separated_text.replace(SentenceSplitter.PLACEHOLDER, '. ')

        return '\n'.join([line.strip() for line in final_text.splitlines() if line.strip()])