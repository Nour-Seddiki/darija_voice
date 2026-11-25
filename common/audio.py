"""Audio helpers: decoding, resampling, and a batched GPU log-mel that reproduces
transformers' WhisperFeatureExtractor._torch_extract_fbank_features exactly
(same STFT, same slaney mel filters, same per-utterance max-8 clamp)."""
import io

import numpy as np
import soundfile as sf
import soxr
import torch

SR = 16000
N_SAMPLES = 30 * SR  # Whisper's fixed 30 s window


def decode_to_16k(audio_bytes: bytes) -> np.ndarray:
    """Any soundfile-readable container (wav/flac/mp3/ogg) -> mono float32 at 16 kHz."""
    x, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        x = soxr.resample(x, sr, SR, quality="HQ")
    return x.astype(np.float32)


def to_int16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767.0).round().astype(np.int16)


class LogMel(torch.nn.Module):
    def __init__(self, feature_extractor, device):
        super().__init__()
        fe = feature_extractor
        self.n_fft, self.hop = fe.n_fft, fe.hop_length
        self.register_buffer("window", torch.hann_window(fe.n_fft, device=device), persistent=False)
        self.register_buffer("mel", torch.from_numpy(fe.mel_filters).to(device, torch.float32), persistent=False)

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, N_SAMPLES) float32 on device -> (B, n_mels, 3000) float32."""
        stft = torch.stft(wav, self.n_fft, self.hop, window=self.window, return_complex=True)
        mag = (stft[..., :-1].abs() ** 2).contiguous()
        mel = self.mel.T @ mag
        log_spec = torch.clamp(mel, min=1e-10).log10()
        max_val = log_spec.amax(dim=(1, 2), keepdim=True)  # per utterance, like the HF batch path
        log_spec = torch.maximum(log_spec, max_val - 8.0)
        return (log_spec + 4.0) / 4.0
