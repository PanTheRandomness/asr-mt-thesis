from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# load model & tokeniser
mt_model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(mt_model_name)
mt_model = AutoModelForSeq2SeqLM.from_pretrained(mt_model_name).to(device)