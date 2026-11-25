"""Qwen2.5-Omni-7B Thinker on an 8 GB GPU.

What is loaded: only the Thinker (audio encoder + Qwen2.5-7B LLM), 4-bit NF4. The Talker /
token2wav (Omni's own voice) are not loaded: the model card documents its speech output only for
Chinese/English, and transformers ships only the code->waveform decoders, not the waveform->code
encoder needed to train that voice on Darija audio. Speech output comes from a separate Darija TTS.

Memory tricks (all reversible, no transformers internals copied):
  * vision encoder deleted after load (audio-only use)
  * 152k x 3584 token embedding (1.1 GB) kept on CPU via CPUEmbedding
  * lm_head wrapped in HeadSwitch: during training the forward returns hidden states and the
    152k-vocab projection is applied only at answer-token positions.
"""
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import HF_CACHE  # noqa: E402

from common.paths import DATA_DIR  # noqa: E402

# Local copy made by `python -m common.robust_download model Qwen/Qwen2.5-Omni-7B` (resumable,
# SHA-256 verified); falls back to the Hub id if it is not there yet.
_LOCAL = DATA_DIR / "models" / "Qwen2.5-Omni-7B"


def _local_complete(d) -> bool:
    import json
    idx = d / "model.safetensors.index.json"
    if not idx.exists():
        return False
    shards = set(json.loads(idx.read_text())["weight_map"].values())
    return all((d / s).exists() for s in shards) and (d / "preprocessor_config.json").exists()


BASE = str(_LOCAL) if _local_complete(_LOCAL) else "Qwen/Qwen2.5-Omni-7B"
SYSTEM = "You are a speech assistant who understands Algerian Darija (الدارجة الجزائرية) perfectly."
ASR_INSTRUCTION = ("Transcribe this audio exactly as spoken, in Algerian Darija using Arabic script "
                   "(keep French words as spoken). Output only the transcription.")


class CPUEmbedding(nn.Module):
    def __init__(self, emb: nn.Embedding, out_device):
        super().__init__()
        self.emb = emb.to("cpu")
        self.out_device = out_device

    @property
    def weight(self):
        return self.emb.weight

    def forward(self, ids):
        return self.emb(ids.to("cpu")).to(self.out_device, non_blocking=True)


class HeadSwitch(nn.Module):
    def __init__(self, head: nn.Linear):
        super().__init__()
        self.head = head
        self.passthrough = False

    @property
    def weight(self):
        return self.head.weight

    def forward(self, h):
        return h if self.passthrough else self.head(h)


def load_processor():
    from transformers import Qwen2_5OmniProcessor
    proc = Qwen2_5OmniProcessor.from_pretrained(BASE, cache_dir=str(HF_CACHE))
    # The stock extractor pads every clip to 300 s (30000 frames). Our clips are <= 30 s and the
    # audio encoder only reads frames marked valid in feature_attention_mask, so pad to 30 s:
    # 10x less CPU for the STFT and 10x smaller tensors, identical encoder input.
    fe = proc.feature_extractor
    fe.chunk_length, fe.n_samples = 30, 30 * fe.sampling_rate
    fe.nb_max_frames = fe.n_samples // fe.hop_length
    return proc


def load_thinker(adapter: str | None = None, trainable: bool = False, device="cuda"):
    from transformers import BitsAndBytesConfig, Qwen2_5OmniThinkerForConditionalGeneration
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             llm_int8_skip_modules=["lm_head", "embed_tokens"])  # visual loads 4-bit (small), then is deleted
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        BASE, cache_dir=str(HF_CACHE), dtype=torch.bfloat16, quantization_config=bnb,
        device_map={"": device}, low_cpu_mem_usage=True)
    model.visual = None  # audio-only: free the vision encoder
    model.model.embed_tokens = CPUEmbedding(model.model.embed_tokens, device)
    model.lm_head = HeadSwitch(model.lm_head)
    torch.cuda.empty_cache()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    return model


LORA_TARGETS = r".*\.(q_proj|k_proj|v_proj|o_proj|out_proj|gate_proj|up_proj|down_proj|fc1|fc2)$"


def build_prompt(processor) -> str:
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "audio", "audio": "placeholder"},
                                         {"type": "text", "text": ASR_INSTRUCTION}]}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


@torch.no_grad()
def transcribe(model, processor, wavs, batch_size=4, max_new_tokens=200, device="cuda"):
    """wavs: list of float32 16 kHz arrays (<= 30 s). Greedy decoding, left-padded batches."""
    head = model.get_base_model().lm_head if hasattr(model, "get_base_model") else model.lm_head
    head.passthrough = False
    prompt = build_prompt(processor)
    order = sorted(range(len(wavs)), key=lambda i: len(wavs[i]))
    out = [None] * len(wavs)
    for s in range(0, len(order), batch_size):
        ids = order[s: s + batch_size]
        enc = processor(text=[prompt] * len(ids), audio=[wavs[i] for i in ids], return_tensors="pt",
                        padding=True, sampling_rate=16000)
        enc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}
        enc["input_features"] = enc["input_features"].to(torch.bfloat16)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                             repetition_penalty=1.0)
        texts = processor.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)
        for j, i in enumerate(ids):
            out[i] = texts[j].strip()
    return out
