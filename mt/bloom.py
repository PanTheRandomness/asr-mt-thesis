import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from typing import List

from utils.model_loader import load_bloom_mt_model
from utils.mt_data_handler import load_data, save_results
from utils.constants import MT_ALLOWED_LANGUAGES, BLOOM_LANG_MAP

MODEL_ID = "bigscience/bloom-560m"

def translate_texts_bloom(
        model,
        tokenizer,
        texts: List[str],
        src_lang: MT_ALLOWED_LANGUAGES,
        tgt_lang: MT_ALLOWED_LANGUAGES,
        batch_size: int = 4
) -> List[str]:
    """
    Translates text with the BLOOM model.

    :param model: Loaded model
    :param tokenizer: Tokenizer
    :param texts: List of source texts
    :param src_lang: Source language (e.g., "fi")
    :param tgt_lang: Target language (ee.g., "en")
    :param batch_size: Batch size for translation
    :return: List of translated texts
    """
    translate_texts = []

    print(f"[MT] Translating from {BLOOM_LANG_MAP.get(src_lang)} to {BLOOM_LANG_MAP.get(tgt_lang)}...")

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            prompts = [
                f"Translate the following {BLOOM_LANG_MAP.get(src_lang)} text to {BLOOM_LANG_MAP.get(tgt_lang)}: '{text}' Translation: "
                for text in batch
            ]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device)

            generated_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + 256,
                num_beams=4,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

            translated_batch = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            cleaned_batch = []
            for original_prompt, generated_text in zip(prompts, translated_batch):
                if generated_text.startswith(original_prompt):
                    translation = generated_text[len(original_prompt):].strip()
                else:
                    translation = generated_text.rsplit("Translation:", 1)[-1].strip()

                translation = translation.split('\n')[0].strip()
                cleaned_batch.append(translation)

            translate_texts.extend(cleaned_batch)

            print(f"✅ Translated {len(translate_texts)} / {len(texts)}.")

        return translate_texts

def main(src_lang: MT_ALLOWED_LANGUAGES, tgt_lang: MT_ALLOWED_LANGUAGES, source_file: str, batch_size: int = 4):
    """
    Main function for BLOOM translation task.
    """

    source_path = os.path.join("..", "data", src_lang, source_file)
    output_dir = os.path.join("..", "results", "mt", MODEL_ID.replace("/", "_"))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"bloom_{src_lang}2{tgt_lang}_results.txt")

    print(f"Loading BLOOM model: {MODEL_ID}...")
    model, tokenizer, device = load_bloom_mt_model(MODEL_ID)

    if model is None or tokenizer is None:
        print("❌ Model loading failed. Terminating.")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        print(f"⚠️ BLOOM Tokenizer_ pad_token set to eos_token ({tokenizer.pad_token}.")

    source_texts = load_data(source_path)

    if not source_texts:
        print("⚠️ No translatable data. Terminating.")
        return

    translated_texts = translate_texts_bloom(model, tokenizer, source_texts, src_lang, tgt_lang, batch_size=batch_size)

    save_results(translated_texts, output_file)

    print("✅ Translation task finished.")