from transformers import BitsAndBytesConfig
import torch
import os

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
QUANTIZATION_CONFIG_4BIT = BitsAndBytesConfig(
    load_in_8bit=False,
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

def load_quantized_model_and_processor(model_class, processor_class, model_name:str, task: str ):
    """
        Common function for loading a model with 4-bit quantization.

        Args:
            model_class: e.g. WhisperForConditionalGeneration, Wav2Vec2ForCTC
            processor_class: e.g. WhisperProcessor, Wav2Vec2Processor
            model_name: name of Hugging Face model (str)
            task: "asr" or "mt"

        Returns:
            Loaded model, processor and device.
        """
    print(f"[{task.upper()}] Loading {model_name} (4-bit) to device: {DEVICE}.")

    # Load Processor / Tokenizer
    try:
        processor = processor_class.from_pretrained(model_name)
    except Exception as e:
        print(f"ERROR loading processor: {e}")
        return None, None, DEVICE

    # Load & Quantize Model
    try:
        model = model_class.from_pretrained(
            model_name,
            quantization_config=QUANTIZATION_CONFIG_4BIT,
            device_map="auto"
        )
    except Exception as e:
        print(f"ERROR loading/quantizing model: {e}")
        model = model_class.from_pretrained(model_name)
        model.to(DEVICE)
        print(f"Returned to non-quantized loading on {DEVICE}.")

    model.eval()
    return model, processor, DEVICE

# --- Specific loading functions ---

def load_whisper_asr_model(model_name: str):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    return load_quantized_model_and_processor(
        WhisperForConditionalGeneration,
        WhisperProcessor,
        model_name,
        "asr"
    )

def load_nemo_asr_model(model_name: str):
    """
    Loads an ASR model from the NVIDIA NeMo registry.

    NOTE: NeMo models are loaded as is. I attempt to use PF16 (Half Precision) on GPU to save VRAM,
    as bnb 4-bit quantisation is not directly supported.

    :param model_name:
    :return:
    """

    global DEVICE

    try:
        from nemo.collections.asr.models import EncDecCTCModel

        print(f"[ASR] Loading NeMo model {model_name} on device: {DEVICE}.")
        model = EncDecCTCModel.from_pretrained(model_name=model_name)

        if DEVICE.startswith("cuda"):
            model = model.to(torch.device(DEVICE))
            print(f"[ASR] Attempting to use FP16 for VRAM saving.")
            model.half()

        model.eval()
        return model, None, DEVICE

    except ImportError:
        print("ERROR: NVIDIA NeMo or its ASR collection is not installed.")
        return None, None, DEVICE
    except Exception as e:
        print(f"ERROR loading NeMo model {model_name}: {e}")
        print("NOTE: NeMo models do not support 4-bit quantisation.")
        return None, None, DEVICE

def load_nllb_mt_model(model_name: str):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    return load_quantized_model_and_processor(
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        model_name,
        "mt"
    )