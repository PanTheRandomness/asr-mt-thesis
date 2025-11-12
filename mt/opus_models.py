import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from typing import List

from utils.model_loader import load_opus_mt_model
from utils.constants import OPUS_MODEL_MAP, MT_ALLOWED_LANGUAGES
from utils.mt_data_handler import load_data, save_results

def get_opus_model_id(src_lang: MT_ALLOWED_LANGUAGES, tgt_lang: MT_ALLOWED_LANGUAGES) -> str:
    """
    Selects the correct Opus-MT model ID based on language pair.

    :param src_lang: Source language code (e.g., "fi")
    :param tgt_lang: Target language core (e.g., "en")
    :return: Hugging Face model ID or None if pair is not supported.
    """

    model_id = OPUS_MODEL_MAP.get((src_lang, tgt_lang))
    if model_id is None:
        print(f"ERROR: Opus-MT model not configured for pair: {src_lang} -> {tgt_lang}.")
    return model_id

def translate_texts_opus(
        model,
        tokenizer,
        texts: List[str],
        src_lang: MT_ALLOWED_LANGUAGES,
        tgt_lang: MT_ALLOWED_LANGUAGES,
        batch_size: int = 8
) -> List[str]:
    """
    Translates source texts using the Helsinki-NLP Opus-MT model.

    :param model: Loaded model
    :param tokenizer: Tokeniser
    :param texts: List of source texts
    :param src_lang: Source language
    :param tgt_lang: Target language
    :param batch_size: Batch size
    :return: Translated texts
    """

    translated_texts = []

    print(f"[MT] Starting translation: {src_lang} -> {tgt_lang} with Opus-MT model: {model}.")

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device)

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512,
                num_beams=5,
                do_sample=False,
            )

            # TODO: What are special tokens & should they be skipped?
            translated_batch = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            translated_texts.extend(translated_batch)

            print(f"Translated {len(translated_texts)} / {len(texts)}.")

        return translated_texts

def main(src_lang: MT_ALLOWED_LANGUAGES, tgt_lang: MT_ALLOWED_LANGUAGES, source_file: str):
    """
    Main function for Opus-MT translation task.
    """

    model_id = get_opus_model_id(src_lang, tgt_lang)
    if not model_id:
        print("Model ID selection failed (Opus-MT). Terminating.")
        return

    source_path = os.path.join("..", "data", src_lang, source_file)
    model_short_name = model_id.split("/")[-1]
    output_dir = os.path.join("..", "data", "results", "mt", model_short_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"opus_{src_lang}2{tgt_lang}_results.txt")

    print(f"Loading Opus-MT model: {model_id}...")
    model, tokenizer, device = load_opus_mt_model(model_id)

    if model is None or tokenizer is None:
        print("Model loading failed. Terminating.")
        return

    source_texts = load_data(source_path)

    if not source_texts:
        print("No translatable data. Terminating.")
        return

    translated_texts = translate_texts_opus(
        model,
        tokenizer,
        source_texts,
        src_lang,
        tgt_lang
    )

    if translated_texts:
        save_results(translated_texts, output_file)
    else:
        print("Translation resulted in no texts. Check for errors.")