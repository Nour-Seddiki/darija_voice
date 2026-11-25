# darija_s2s — Darija + French + Arabic speech-to-speech on Qwen2.5-Omni-7B

Goal: a voice model you talk to in Algerian Darija, French or Arabic (MSA) that answers in speech
in the same language, built on an 8 GB RTX 5050 laptop (plus rented GPUs for the big voice runs).

## Plan (trilingual)

```
 dz / fr / ar speech ──► Omni-7B Thinker (4-bit + LoRA) ──► hidden states + text
                                                                │
                                   own Talker (LLM) ──► open-codec tokens ──► codec decoder ──► speech
```

| phase | what | status |
|---|---|---|
| 1 | Omni Thinker learns to hear Darija (then fr/ar) | code ready, waiting for model download |
| 2 | pick the open codec (Mimi / SNAC) by round-trip tests on dz/fr/ar clips — `codec/bench_codecs.py` | code ready |
| 3 | train the voice (Talker) on codec tokens of dz/fr/ar data | next |
| 4 | condition the Talker on Thinker hidden states (one model, low latency) | later |
| 5 | dialogue data (generated Darija conversations, voiced by our TTS) | later |

Omni's own Talker cannot be retrained (no public speech encoder), so the voice uses an open codec
whose encoder IS public: `kyutai/mimi` (CC-BY-4.0, 12.5 Hz x up to 32 codebooks, Moshi's codec) or
`hubertsiuzdak/snac_24khz` (MIT, ~84 tokens/s, Orpheus' codec).

## Architecture

```
 Darija speech ──► Qwen2.5-Omni-7B Thinker (4-bit + LoRA) ──► Darija text ──► Darija TTS ──► speech
                   audio encoder + Qwen2.5-7B LLM                              (stage 3)
```

* **Base model: `Qwen/Qwen2.5-Omni-7B`**, Thinker only (audio encoder + LLM), loaded 4-bit NF4,
  fine-tuned with LoRA on both the LLM and the audio encoder (`omni/`).
* **Omni's built-in voice (Talker + token2wav) is not used for Darija.** Its model card documents
  speech output only for Chinese/English (2 voices, Seed-TTS-eval), and transformers ships only the
  code→waveform decoders (DiT + BigVGAN). The waveform→code encoder needed to *train* the Talker on
  Darija audio is not released (checked in transformers and in both the 3B and 7B repos: only
  `spk_dict.pt`, the two stock voices), so the voice comes from a separate Darija TTS (stage 3).
  Darija *text* tokenizes fine with Qwen's tokenizer; only speech codes are unavailable.
  Because the 3B offers no voice-training advantage, the project stays on 7B (decided 2026-11-25).
* Fits 8 GB by: deleting the vision encoder, keeping the 1.1 GB token embedding on CPU, and
  applying the 152k-vocab `lm_head` only at answer positions (`omni/model.py`).

## Data audit (2026-11-25) — what is really Algerian

| source | HF dataset | verdict | use |
|---|---|---|---|
| kahwa | oddadmix/arabic-audio-collection-algerian-kahwa-postcast | Algerian, 110 h, podcast, FR code-switching | train |
| rawi | oddadmix/arabic-audio-collection-algerian-rawi | Algerian, 51 h, stories (many clips > 30 s are dropped) | train |
| loubna | oddadmix/arabic-audio-collection-algerian-loubna-stories | Algerian, 236 h, **one speaker** | train (capped 60 h); TTS voice |
| yasmine | yasminekaced/algerian_tts | Algerian, 34 h, YouTube **rolling captions** (raw clips overlap the next caption by ~2.5 s — re-stitched in prep) | train |
| cafe | 182 CAFE rows of quantumpuls3/Algerian-STT-Super-Dataset-V2 | Algerian code-switching | train |
| casablanca_val / _test | UBC-NLP/Casablanca (Algeria) | human-labelled, CC-BY-NC-ND | **eval only** |
| ✗ | quantumpuls3/Algerian-STT-Super-Dataset-V2 (rest) | "Darija" rows are **Egyptian**, "Adiren" **Moroccan**, "SADA22" **Saudi**, and it contains the Casablanca test set | not used |
| ✗ | quantumpuls3/Algerian-STT-Cleaned-V5 | Moroccan ("Adiren") | not used |

French and Arabic (MSA) sources:

| source | HF dataset | lang | notes |
|---|---|---|---|
| mls_fr | facebook/multilingual_librispeech (french) | fr | CC-BY-4.0, audiobooks, many speakers, lowercase/no punctuation; capped at 4 shards (~125 h) |
| siwis_fr | Rcarvalo/siwis-fr-24k | fr | CC-BY-4.0, one studio speaker, punctuated — TTS quality |
| cv_ar | fixie-ai/common_voice_17_0 (ar) | ar | CC0, many read-speech speakers, MSA (partly diacritised) |
| clartts_ar | MBZUAI/ClArTTS | ar | CC-BY-4.0, one studio speaker, fully diacritised Classical Arabic; audio stored as float64 arrays (13.5 GB) |
| ✗ | MohamedRashad/MASC-Arabic | — | not used for MSA: contains Egyptian and other dialects |

Every prepared clip carries `lang` = `dz` / `fr` / `ar`. Prepared audio lives outside OneDrive in
`C:\Users\seddi\darija_data\prepared\<source>\` (16 kHz int16 `.bin` + `.jsonl` index per parquet shard).

Downloads use `common/robust_download.py` (byte-range parts, resume across restarts, SHA-256 check)
because this connection resets every few minutes and huggingface_hub discards partial files on
retry. `fetch_queue.ps1` runs the long queue (Omni-7B, then the rest of the data) detached.

## Baselines on Casablanca-Algeria validation (300 clips, seed 0)

| model | WER | CER |
|---|---|---|
| openai/whisper-large-v3-turbo (no fine-tune) | 0.919 | 0.444 |
| Qwen2.5-Omni-7B Thinker 4-bit (no fine-tune) | see `omni.eval_omni` run | |

WER/CER use `normalize_for_wer` on both sides (diacritics, punctuation, hamza/alef, ta-marbuta,
alef-maqsura folded), so they measure recognition, not spelling convention.

## Run it (from this folder, with the deep_learning `.venv`)

```powershell
$py = "..\.venv\Scripts\python.exe"
& $py -m asr.prepare_data --sources casablanca_val casablanca_test cafe kahwa rawi yasmine loubna
& $py -m common.robust_download model Qwen/Qwen2.5-Omni-7B
& $py -m omni.eval_omni --limit 300                       # baseline
& $py -m omni.train_omni --run smoke --smoke 4            # memory / speed check
& $py -m omni.train_omni --run omni7b_asr_v1              # stage 1 (resumable: add --resume)
& $py -m omni.eval_omni --adapter C:\Users\seddi\darija_data\runs\omni7b_asr_v1\best --split casablanca_test
```

Training scripts pin themselves to P-cores, keep the laptop out of Modern Standby, and cap the
CUDA cache (`common/winrun.py`).

## Stages

1. **Listening (now):** Darija audio → Darija text. Teaches Omni Darija phonetics/vocabulary.
2. **Talking back:** Darija audio → Darija *reply*. No Algerian dialogue dataset exists, so replies
   must be built (e.g. generated for the stage-1 transcripts, then filtered) before training.
3. **Voice:** fine-tune a TTS (e.g. `facebook/mms-tts-ara`, VITS, 16 kHz, plain Arabic letters,
   no ڨ/diacritics → `normalize_target`) on Loubna's single-speaker 236 h, then chain it after Omni.

`asr/train_whisper.py` (Whisper-turbo + LoRA) is kept as a comparison baseline; it was smoke-tested
(5.7 s/step at batch 8, 5.3 GB) but not trained.
