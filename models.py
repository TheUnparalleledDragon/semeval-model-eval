"""Model-specific loaders behind one image/question -> answer interface."""

from dataclasses import dataclass
from importlib import metadata
from contextlib import contextmanager, nullcontext

MODEL_ADAPTER_VERSION = 2


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    backend: str = "multimodal"
    trust_remote_code: bool = False


MODELS = {
    "internvl3_5_4b": ModelSpec("OpenGVLab/InternVL3_5-4B-HF"),
    "qwen3_vl_8b": ModelSpec("Qwen/Qwen3-VL-8B-Instruct"),
    "aya_vision_8b": ModelSpec("CohereLabs/aya-vision-8b"),
    "culturalpangea_7b": ModelSpec("neulab/CulturalPangea-7B", backend="pangea"),
    "minicpm_o_4_5": ModelSpec("openbmb/MiniCPM-o-4_5", backend="minicpm_o", trust_remote_code=True),
    "minicpm_v_4_5": ModelSpec("openbmb/MiniCPM-V-4_5", backend="minicpm_v", trust_remote_code=True),
    "gemma_4_12b": ModelSpec("google/gemma-4-12B-it", backend="multimodal"),
    "qwen3_8_27b": ModelSpec("Qwen/Qwen3.8-27B", backend="multimodal"),
}

# Quantize language layers, retaining the vision encoder and connector in the
# selected floating precision. Custom skip lists must also retain lm_head.
VISION_MODULES = {
    "internvl3_5_4b": ("vision_tower", "multi_modal_projector"),
    "qwen3_vl_8b": ("visual",),
    "aya_vision_8b": ("vision_tower", "multi_modal_projector"),
    "culturalpangea_7b": ("vision_tower", "mm_projector"),
    "minicpm_o_4_5": ("vpm", "resampler"),
    "minicpm_v_4_5": ("vpm", "resampler"),
    "gemma_4_12b": ("embed_vision",),
    "qwen3_8_27b": ("visual",),
}


def cuda_compute_dtype():
    """Use native BF16 only when every GPU in device_map='auto' supports it."""
    import torch

    native_bf16 = all(torch.cuda.get_device_capability(i)[0] >= 8
                      for i in range(torch.cuda.device_count()))
    return torch.bfloat16 if native_bf16 else torch.float16


def quantization_config(key, dtype):
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
        llm_int8_skip_modules=["lm_head", *VISION_MODULES[key]],
    )


@contextmanager
def inference_context(dtype):
    import torch

    autocast = (torch.autocast("cuda", dtype=dtype) if torch.cuda.is_available()
                else nullcontext())
    with torch.inference_mode(), autocast:
        yield


def clean_answer(value):
    """Normalize transport shapes, retaining the model's words."""
    if isinstance(value, list):
        if not value:
            return ""
        value = value[0]
    if isinstance(value, dict):
        value = value.get("generated_text", value.get("text", ""))
    if isinstance(value, list):
        value = value[-1].get("content", "") if value else ""
    if isinstance(value, list):
        value = " ".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in value)
    return str(value).strip()


def validate_quantization_dependency():
    """Fail before downloading model assets when Kaggle's bnb is too old."""
    from packaging.version import Version

    try:
        installed = metadata.version("bitsandbytes")
    except metadata.PackageNotFoundError:
        installed = None
    if installed is None or Version(installed) < Version("0.46.1"):
        raise RuntimeError(
            f"4-bit loading requires bitsandbytes>=0.46.1; installed: {installed or 'missing'}. "
            "In a Kaggle notebook cell run: %pip install --upgrade 'bitsandbytes>=0.46.1'. "
            "Restart the notebook kernel if these libraries were already imported, then rerun."
        )


def validate_transformers_dependency(key):
    """Verify released library support before loading checkpoint assets."""
    from packaging.version import Version
    import transformers

    repair = (
        f"Loaded Transformers {transformers.__version__} from {transformers.__file__}. "
        "In Kaggle run: %pip install --upgrade 'transformers>=5.13.0,<6' "
        "'bitsandbytes>=0.46.1' accelerate. Restart the notebook kernel if Transformers "
        "was already imported, then rerun the token setup and inference cells."
    )
    if Version(transformers.__version__) < Version("5.13.0"):
        raise RuntimeError("This pipeline requires Transformers>=5.13.0. " + repair)
    architecture = {
        "gemma_4_12b": "gemma4_unified", "qwen3_8_27b": "qwen3_5",
        "qwen3_vl_8b": "qwen3_vl", "internvl3_5_4b": "internvl",
        "aya_vision_8b": "aya_vision",
    }.get(key)
    if architecture:
        try:
            transformers.AutoConfig.for_model(architecture)
        except (ValueError, KeyError, AttributeError) as exc:
            raise RuntimeError(f"This Transformers installation does not recognize {key}'s "
                               f"{architecture} architecture. " + repair) from exc


def load_model(key, four_bit=True):
    if key not in MODELS:
        raise ValueError(f"Unknown model {key!r}. Choose from: {', '.join(MODELS)}")
    spec = MODELS[key]
    validate_transformers_dependency(key)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. In Kaggle, enable a GPU accelerator.")
    if four_bit:
        validate_quantization_dependency()
    dtype = cuda_compute_dtype()
    print(f"Loading {key}: compute dtype={dtype}, language 4-bit={four_bit}; "
          "vision modules kept in floating precision", flush=True)
    if spec.backend == "pangea":
        try:
            from llava.model.builder import load_pretrained_model
        except ImportError as exc:
            raise RuntimeError(
                "CulturalPangea requires LLaVA-NeXT. Install it as described in README.md."
            ) from exc
        # Bypass the builder's load_4bit shortcut: it hardcodes FP16 and passes
        # load_in_4bit (removed from modern Transformers). Supply our config.
        extra = {"quantization_config": quantization_config(key, dtype)} if four_bit else {}
        tokenizer, model, processor, _ = load_pretrained_model(
            spec.repo, None, "CulturalPangea-7B-qwen", multimodal=True,
            torch_dtype=str(dtype).split(".")[-1], attn_implementation="sdpa", **extra,
        )
        return PangeaRunner(tokenizer, model.eval(), processor, dtype)

    if spec.backend.startswith("minicpm"):
        from transformers import AutoModel, AutoTokenizer

        kwargs = {"trust_remote_code": True, "device_map": "auto", "torch_dtype": dtype,
                  "attn_implementation": "sdpa"}
        if spec.backend == "minicpm_o":
            kwargs.update(init_vision=True, init_audio=False, init_tts=False)
        if four_bit:
            kwargs["quantization_config"] = quantization_config(key, dtype)
        model = AutoModel.from_pretrained(spec.repo, **kwargs).eval()
        tokenizer = AutoTokenizer.from_pretrained(spec.repo, trust_remote_code=True)
        return MiniCPMRunner(model, tokenizer, spec.backend, dtype)

    model_kwargs = {"device_map": "auto", "dtype": dtype}
    if four_bit:
        model_kwargs["quantization_config"] = quantization_config(key, dtype)
    if spec.backend == "multimodal":
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(spec.repo, trust_remote_code=spec.trust_remote_code)
        model = AutoModelForMultimodalLM.from_pretrained(
            spec.repo, trust_remote_code=spec.trust_remote_code, **model_kwargs).eval()
        return MultimodalRunner(model, processor, key == "gemma_4_12b", dtype)
    raise ValueError(f"Unsupported backend: {spec.backend}")


class MiniCPMRunner:
    def __init__(self, model, tokenizer, backend, dtype):
        self.model, self.tokenizer, self.backend = model, tokenizer, backend
        self.dtype = dtype

    def answer(self, image, prompt, max_new_tokens):
        kwargs = {"msgs": [{"role": "user", "content": [image, prompt]}],
                  "tokenizer": self.tokenizer, "max_new_tokens": max_new_tokens,
                  "enable_thinking": False, "stream": False}
        if self.backend == "minicpm_o":
            kwargs.update(use_tts_template=False, generate_audio=False, do_sample=False,
                          num_beams=1, repetition_penalty=1.0)
        else:
            # MiniCPM-V ignores do_sample in **kwargs. Its actual switch is sampling.
            kwargs.update(sampling=False, num_beams=1, repetition_penalty=1.0)
        with inference_context(self.dtype):
            return clean_answer(self.model.chat(**kwargs))


class MultimodalRunner:
    def __init__(self, model, processor, gemma, dtype):
        self.model, self.processor, self.gemma = model, processor, gemma
        self.dtype = dtype

    def answer(self, image, prompt, max_new_tokens):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}
        ]}]
        options = {"tokenize": True, "return_dict": True,
                   "return_tensors": "pt", "add_generation_prompt": True,
                   "enable_thinking": False}
        # BatchFeature.to casts floating tensors only: IDs/masks/grid positions
        # keep their integer types. Never cast a quantized model with model.to().
        inputs = self.processor.apply_chat_template(messages, **options).to(
            self.model.device, dtype=self.dtype)
        length = inputs["input_ids"].shape[-1]
        with inference_context(self.dtype):
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        tokens = output[0][length:]
        if self.gemma:
            raw = self.processor.decode(tokens, skip_special_tokens=False)
            parsed = self.processor.parse_response(raw, prefix=inputs["input_ids"][0])
            return clean_answer(parsed.get("content", ""))
        return self.processor.decode(tokens, skip_special_tokens=True).strip()


class PangeaRunner:
    def __init__(self, tokenizer, model, image_processor, dtype):
        self.tokenizer, self.model, self.image_processor = tokenizer, model, image_processor
        self.dtype = dtype

    def answer(self, image, prompt, max_new_tokens):
        import torch
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import tokenizer_image_token, process_images

        # The checkpoint card uses Qwen chat markers and an image token.
        text = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<image>\n" + prompt +
                "<|im_end|>\n<|im_start|>assistant\n")
        ids = tokenizer_image_token(
            text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)
        # Respect the checkpoint's any-resolution/padding preprocessing and the
        # vision tower's own device/precision, which may differ from the LLM.
        vision = self.model.get_vision_tower()
        pixels = process_images([image], self.image_processor, self.model.config)
        if isinstance(pixels, list):
            pixels = [value.to(vision.device, dtype=vision.dtype) for value in pixels]
        else:
            pixels = pixels.to(vision.device, dtype=vision.dtype)
        with inference_context(self.dtype):
            output = self.model.generate(
                ids, images=pixels, image_sizes=[image.size],
                do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
            )
        # LLaVA generate normally returns new tokens only; trim the input if present.
        if output.shape[-1] >= ids.shape[-1] and torch.equal(output[0, :ids.shape[-1]], ids[0]):
            output = output[:, ids.shape[-1]:]
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
