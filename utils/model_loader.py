from numpy.core.defchararray import startswith
from transformers import BitsAndBytesConfig
import torch

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
    """
    Loads Whisper Large with 4-bit quantisation.

    :param model_name: Model name
    :return: quantised model & processor
    """
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

    :param model_name: Model name
    :return: Model, processor, device
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

def load_wav2vec2_asr_model(model_name: str):
    """
    Loads Wav2Vec2 model by constructing the Processor from Auto components.

    :param model_name: Model name
    :return: Quantised model & processor
    """
    from transformers import AutoFeatureExtractor, AutoTokenizer, Wav2Vec2Processor, Wav2Vec2ForCTC
    import torch
    import traceback

    global DEVICE

    print(f"[ASR] Loading {model_name} (FP16/FP32 fallback) to device: {DEVICE}. Robust Auto construction.")

    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        print("✅ Feature Extractor loaded.")
    except Exception as e:
        print(f"ERROR loading AutoFeatureExtractor: {e}")
        traceback.print_exc()
        return None, None, DEVICE

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ AutoTokenizer loaded.")
    except Exception as e:
        print(f"FATAL ERROR loading AutoTokenizer: {e}")
        traceback.print_exc()
        return None, None, DEVICE

    try:
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        print("✅ Processor constructed successfully from components.")
    except Exception as e:
        print(f"ERROR constructing Wav2Vec2Processor: {e}")
        traceback.print_exc()
        return None, None, DEVICE

    try:
        model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            dtype=torch.float16 if DEVICE.startswith("cuda") else torch.float32
        )
        model.to(DEVICE)
        print(f"✅ Loaded Wav2Vec2ForCTC model non-quantised/FP16 on {DEVICE}.")
    except Exception as e:
        print(f"ERROR loading model {model_name}: {e}")
        traceback.print_exc()
        return None, None, DEVICE

    model.eval()
    return model, processor, DEVICE

def load_nllb_mt_model(model_name: str):
    """
    Loads NLLB model, prioritising 8-bit quantisation for VRAM efficiency on CUDA.

    :param model_name: Model name
    :return: Quantised model, tokeniser & device
    """

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_class = AutoModelForSeq2SeqLM
    tokenizer_class = AutoTokenizer
    task = "mt"
    print(f"[{task.upper()}] Loading {model_name} (8-bit preferred) to device: {DEVICE}.")

    try:
        tokenizer = tokenizer_class.from_pretrained(model_name)
    except Exception as e:
        print(f"❌ ERROR loading tokenizer: {e}")
        return None, None, DEVICE

    model = None
    if DEVICE.startswith("cuda"):
        try:
            model = model_class.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto"
            )
            print(f"✅ Successfully loaded NLLB in 8-bit quantisation mode.")
        except Exception as e:
            print(f"❌ ERROR: 8-bit loading failed for NLLB: {e}. Attempting fallback non-quantised loading.")

    if model is None:
        try:
            model = model_class.from_pretrained(
                model_name,
                dtype=torch.float16 if DEVICE.startswith("cuda") else torch.float32
            )
            model.to(DEVICE)
            print(f"✅ Loaded NLLB model non-quantised on {DEVICE}.")
        except Exception as e:
            print(f"❌ ERROR: Fallback loading failed. Model may be too large for {DEVICE}: {e}")
            return None, None, DEVICE

    model.eval()
    return model, tokenizer, DEVICE

def load_bloom_mt_model(model_name: str):
    """
    Loads BLOOM model with 4-bit quantisation.

    :param model_name: Model name
    :return: Quantised model & tokeniser
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    return load_quantized_model_and_processor(
        AutoModelForCausalLM,
        AutoTokenizer,
        model_name,
        "mt"
    )

def load_opus_mt_model(model_name: str):
    """
    Loads Helsinki-NLP Opus-MT model (Seq2Seq).

    :param model_name: Model name from Hugging Face (e.g., "Helsinki-NLP/opus-mt-en-fi")
    :return: Loaded model, tokeniser, and device.
    """

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    print(f"[MT] Loading {model_name} to device: {DEVICE}.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"❌ ERROR loading tokenizer for {model_name}: {e}.")
        return None, None, DEVICE

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            dtype=torch.float16 if DEVICE.startswith("cuda") else torch.float32
        )
        model.to(DEVICE)
        print(f"✅ Loaded Opus-MT model non-quantised on {DEVICE} (Dtype: {model.dtype}.")
    except Exception as e:
        print(f"❌ ERROR loading model {model_name}: {e}.")
        return None, None, DEVICE

    model.eval()
    return model, tokenizer, DEVICE