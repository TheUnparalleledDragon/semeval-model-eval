"""Model-specific loaders behind one image/question -> answer interface."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    backend: str = "pipeline"
    trust_remote_code: bool = False


MODELS = {
    "internvl3_5_4b": ModelSpec("OpenGVLab/InternVL3_5-4B", trust_remote_code=True),
    "qwen3_vl_8b": ModelSpec("Qwen/Qwen3-VL-8B-Instruct"),
    "aya_vision_8b": ModelSpec("CohereLabs/aya-vision-8b"),
    "culturalpangea_7b": ModelSpec("neulab/CulturalPangea-7B", backend="pangea"),
    "minicpm_o_4_5": ModelSpec("openbmb/MiniCPM-o-4_5", backend="minicpm_o", trust_remote_code=True),
    "minicpm_v_4_5": ModelSpec("openbmb/MiniCPM-V-4_5", backend="minicpm_v", trust_remote_code=True),
    "gemma_4_12b": ModelSpec("google/gemma-4-12B-it", backend="multimodal"),
    "qwen3_8_27b": ModelSpec("Qwen/Qwen3.8-27B", backend="multimodal"),
}


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


def load_model(key, four_bit=True):
    if key not in MODELS:
        raise ValueError(f"Unknown model {key!r}. Choose from: {', '.join(MODELS)}")
    spec = MODELS[key]
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. In Kaggle, enable a GPU accelerator.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if spec.backend == "pangea":
        try:
            from llava.model.builder import load_pretrained_model
        except ImportError as exc:
            raise RuntimeError(
                "CulturalPangea requires LLaVA-NeXT. Install it as described in README.md."
            ) from exc
        tokenizer, model, processor, _ = load_pretrained_model(
            spec.repo, None, "CulturalPangea-7B-qwen", multimodal=True,
            load_4bit=four_bit, attn_implementation="sdpa",
        )
        return PangeaRunner(tokenizer, model.eval(), processor)

    if spec.backend.startswith("minicpm"):
        from transformers import AutoModel, AutoTokenizer

        kwargs = {"trust_remote_code": True, "device_map": "auto", "torch_dtype": dtype}
        if spec.backend == "minicpm_o":
            kwargs.update(init_vision=True, init_audio=False, init_tts=False)
        if four_bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
            )
        model = AutoModel.from_pretrained(spec.repo, **kwargs).eval()
        tokenizer = AutoTokenizer.from_pretrained(spec.repo, trust_remote_code=True)
        return MiniCPMRunner(model, tokenizer, spec.backend)

    model_kwargs = {"device_map": "auto", "torch_dtype": dtype}
    if four_bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
        )
    if spec.backend == "multimodal":
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(spec.repo)
        model = AutoModelForMultimodalLM.from_pretrained(spec.repo, **model_kwargs).eval()
        return MultimodalRunner(model, processor, key == "gemma_4_12b")

    from transformers import pipeline

    pipe = pipeline(
        "image-text-to-text", model=spec.repo, trust_remote_code=spec.trust_remote_code,
        model_kwargs=model_kwargs,
    )
    return PipelineRunner(pipe)


class PipelineRunner:
    def __init__(self, pipe):
        self.pipe = pipe

    def answer(self, image, prompt, max_new_tokens):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}
        ]}]
        result = self.pipe(
            text=messages, max_new_tokens=max_new_tokens,
            do_sample=False, return_full_text=False,
        )
        return clean_answer(result)


class MiniCPMRunner:
    def __init__(self, model, tokenizer, backend):
        self.model, self.tokenizer, self.backend = model, tokenizer, backend

    def answer(self, image, prompt, max_new_tokens):
        kwargs = {"msgs": [{"role": "user", "content": [image, prompt]}],
                  "tokenizer": self.tokenizer, "max_new_tokens": max_new_tokens,
                  "do_sample": False, "enable_thinking": False}
        if self.backend == "minicpm_o":
            kwargs["use_tts_template"] = False
        return clean_answer(self.model.chat(**kwargs))


class MultimodalRunner:
    def __init__(self, model, processor, gemma):
        self.model, self.processor, self.gemma = model, processor, gemma

    def answer(self, image, prompt, max_new_tokens):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}
        ]}]
        options = {"tokenize": True, "return_dict": True,
                   "return_tensors": "pt", "add_generation_prompt": True,
                   "enable_thinking": False}
        inputs = self.processor.apply_chat_template(messages, **options).to(self.model.device)
        length = inputs["input_ids"].shape[-1]
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        tokens = output[0][length:]
        if self.gemma:
            raw = self.processor.decode(tokens, skip_special_tokens=False)
            parsed = self.processor.parse_response(raw, prefix=inputs["input_ids"][0])
            return clean_answer(parsed.get("content", ""))
        return self.processor.decode(tokens, skip_special_tokens=True).strip()


class PangeaRunner:
    def __init__(self, tokenizer, model, image_processor):
        self.tokenizer, self.model, self.image_processor = tokenizer, model, image_processor

    def answer(self, image, prompt, max_new_tokens):
        import torch
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import tokenizer_image_token

        # The checkpoint card uses Qwen chat markers and an image token.
        text = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<image>\n" + prompt +
                "<|im_end|>\n<|im_start|>assistant\n")
        ids = tokenizer_image_token(
            text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)
        pixels = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
        pixels = pixels.to(self.model.device, dtype=torch.float16)
        with torch.inference_mode():
            output = self.model.generate(
                ids, images=[pixels], image_sizes=[image.size],
                do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
            )
        # LLaVA generate normally returns new tokens only; trim the input if present.
        if output.shape[-1] >= ids.shape[-1] and torch.equal(output[0, :ids.shape[-1]], ids[0]):
            output = output[:, ids.shape[-1]:]
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
