import torch
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils.model_loader import load_nllb_mt_model
from utils.mt_data_handler import load_data, save_results
from typing import List
from utils.constants import NLLB_LANG_MAP, MT_ALLOWED_LANGUAGES

MODEL_ID = "facebook/nllb-200-distilled-600M"

def translate_texts_nllb(
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        texts: List[str],
        src_lang: MT_ALLOWED_LANGUAGES,
        tgt_lang: MT_ALLOWED_LANGUAGES,
        batch_size: int = 4
) -> List[str]:
    """
    Translates source texts.

    :param model: Model
    :param tokenizer: Tokeniser
    :param texts: List of source texts
    :param src_lang: Source language
    :param tgt_lang: Target language
    :param batch_size: Batch size
    :return: Translated texts
    """

    src_code = NLLB_LANG_MAP.get(src_lang, src_lang)
    tgt_code = NLLB_LANG_MAP.get(tgt_lang, tgt_lang)

    try:
        target_id = tokenizer.lang_code_to_id[tgt_code]
    except KeyError:
        print(f"ERROR: Target language :{tgt_lang} ({tgt_code}) is not supported by model :{MODEL_ID}.")
        return []

    print(f"Starting translation: {src_lang} ({src_code}) -> {tgt_lang} ({tgt_code}).")

    translated_texts = []
    model.eval()

    tokenizer.src_lang = src_code
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)

            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device)

            generated_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                forced_bos_token_id=target_id,
                max_length=512,
                num_beams=4,
                do_sample=False,
            )

            translated_batch = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            translated_texts.extend(translated_batch)

            print(f"Translated {len(translated_texts)} / {len(texts)}")

    return translated_texts


def main(src_lang: MT_ALLOWED_LANGUAGES, tgt_lang: MT_ALLOWED_LANGUAGES, source_file: str):

    source_path = os.path.join("..", "data", src_lang, source_file)
    output_dir = os.path.join("..", "results", "mt", MODEL_ID.replace("/", "_"))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"nllb_{src_lang}2{tgt_lang}_results.txt")

    model, tokenizer, device = load_nllb_mt_model(MODEL_ID)

    if model is None or tokenizer is None:
        print("Model loading failed. Terminating.")
        return

    source_texts = load_data(source_path)

    if not source_texts:
        print("No translatable data. Terminating.")
        return

    translated_texts = translate_texts_nllb(model, tokenizer, source_texts, src_lang, tgt_lang, batch_size=4)
    save_results(translated_texts, output_file)

    print(f"NLLB Translation ({src_lang}->{tgt_lang}) complete. Saved to file: {output_file}")