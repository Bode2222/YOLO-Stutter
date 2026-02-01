import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import utils.vits.commons as commons
import utils.vits.utils as utils
from utils.vits.models import SynthesizerTrn
from utils.vits.text.symbols import symbols
from utils.vits.text import text_to_sequence, cleaners
from utils.vits.mel_processing import spectrogram_torch

TARGET_SAMPLE_RATE = 22050
FILTER_LENGTH = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
MAX_SPEC_FRAMES = 1024
MAX_TEXT_TOKENS = 768
LABELS = ["rep", "block", "missing", "replace", "prolong"]


def load_audio_mono(filename, target_sample_rate=TARGET_SAMPLE_RATE):
    try:
        import torchaudio
        import torchaudio.transforms as T

        waveform, sample_rate = torchaudio.load(filename)
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        elif waveform.ndim == 2:
            waveform = waveform.squeeze(0)

        if sample_rate != target_sample_rate:
            resampler = T.Resample(sample_rate, target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = target_sample_rate

        return waveform.to(torch.float32), sample_rate
    except Exception:
        import librosa

        audio, sample_rate = librosa.load(filename, sr=target_sample_rate, mono=True)
        return torch.from_numpy(audio).to(torch.float32), sample_rate


def compute_spec(audio, sampling_rate=TARGET_SAMPLE_RATE):
    max_wav_value = 32768.0
    peak = torch.max(torch.abs(audio))
    if peak > 0:
        normalized_waveform = audio / peak
    else:
        normalized_waveform = audio

    audio = normalized_waveform * 32767
    audio_norm = audio / max_wav_value
    audio_norm = audio_norm.unsqueeze(0)

    spec = spectrogram_torch(
        audio_norm,
        FILTER_LENGTH,
        sampling_rate,
        HOP_LENGTH,
        WIN_LENGTH,
        center=False,
    )
    spec = torch.squeeze(spec, 0)
    return spec


def safe_text_to_sequence(text, cleaner_names):
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    try:
        for name in cleaner_names:
            cleaner = getattr(cleaners, name)
            if not cleaner:
                raise Exception(f"Unknown cleaner: {name}")
            text = cleaner(text)
    except RuntimeError as exc:
        if "espeak" not in str(exc).lower():
            raise
        text = cleaners.basic_cleaners(text)

    text = "".join(ch if ch in symbol_to_id else " " for ch in text)
    text = re.sub(r"\s+", " ", text).strip()
    return [symbol_to_id[ch] for ch in text] if text else []


def get_text(text, hps):
    try:
        text_norm = text_to_sequence(text, hps.data.text_cleaners)
    except RuntimeError as exc:
        if "espeak" in str(exc).lower():
            text_norm = safe_text_to_sequence(text, ["basic_cleaners"])
        else:
            raise
    except KeyError:
        text_norm = safe_text_to_sequence(text, hps.data.text_cleaners)

    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    return torch.LongTensor(text_norm)


def get_soft_attention(net_g, text, text_lengths, spec, spec_lengths):
    with torch.no_grad():
        _, _, (neg_cent, _), _, _, _, _ = net_g(
            text, text_lengths, spec, spec_lengths
        )
        neg_cent = nn.functional.softmax(neg_cent, dim=-1)
    return neg_cent


def single_inference_from_spec(hps, net_g, spec, ref_text, downsample_factor, decoder, device):
    text = get_text(ref_text, hps)
    text_length = torch.tensor(text.size(0))
    spec_length = torch.tensor(spec.shape[-1])

    text = text.unsqueeze(0).to(device)
    spec = spec.unsqueeze(0).to(device)
    text_length = text_length.unsqueeze(0).to(device)
    spec_length = spec_length.unsqueeze(0).to(device)

    num_regions = int(spec_length // downsample_factor)

    soft_attention = get_soft_attention(net_g, text, text_length, spec, spec_length)
    new_soft_attention = nn.functional.pad(
        soft_attention,
        (
            0,
            MAX_TEXT_TOKENS - soft_attention.shape[-1],
            0,
            MAX_SPEC_FRAMES - soft_attention.shape[-2],
        ),
    )

    mask = torch.ones((1, 64), dtype=torch.bool)
    mask[0, : num_regions + 1] = False
    return decoder(new_soft_attention, mask.to(device))


def chunk_spec(spec, max_frames=MAX_SPEC_FRAMES):
    total_frames = spec.shape[-1]
    chunks = []
    for start in range(0, total_frames, max_frames):
        end = start + max_frames
        chunks.append((spec[:, start:end], start))
    return chunks


def split_text_by_duration(text, durations_sec):
    if not durations_sec:
        return [text]
    words = text.split()
    if not words:
        return [""] * len(durations_sec)

    total_duration = sum(durations_sec)
    if total_duration <= 0:
        total_duration = len(durations_sec)

    words_per_sec = len(words) / total_duration if total_duration else len(words)
    chunks = []
    idx = 0
    remaining_words = len(words)

    for i, duration in enumerate(durations_sec):
        remaining_chunks = len(durations_sec) - i
        if remaining_words == 0:
            chunks.append("")
            continue

        if i == len(durations_sec) - 1:
            count = remaining_words
        else:
            count = int(round(duration * words_per_sec))
            count = max(1, count) if remaining_words >= remaining_chunks else 1
            max_for_chunk = max(1, remaining_words - (remaining_chunks - 1))
            count = min(count, max_for_chunk)

        chunks.append(" ".join(words[idx : idx + count]))
        idx += count
        remaining_words -= count

    return chunks


def truncate_text_to_tokens(text, hps, max_tokens=MAX_TEXT_TOKENS):
    if not text:
        return text
    if len(get_text(text, hps)) <= max_tokens:
        return text

    words = text.split()
    current = []
    for word in words:
        candidate = " ".join(current + [word])
        if len(get_text(candidate, hps)) > max_tokens:
            break
        current.append(word)

    if not current:
        return words[0]
    return " ".join(current)


def main():
    parser = argparse.ArgumentParser(description="Minimal YOLO-Stutter inference")
    parser.add_argument("--audio", required=True, help="Path to input wav")
    parser.add_argument("--transcript", required=True, help="Path to transcript text")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "utils" / "vits" / "configs" / "ljs_base.json"))
    parser.add_argument("--model", default=str(PROJECT_ROOT / "saved_models" / "pretrained_ljs.pth"))
    parser.add_argument("--decoder", default=str(PROJECT_ROOT / "saved_models" / "decoder_tts_joint"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--downsample-factor", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", help="Optional output JSON path")
    args = parser.parse_args()

    hps = utils.get_hparams_from_file(args.config)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).eval()
    _ = utils.load_checkpoint(args.model, net_g, None)

    device = torch.device(args.device)
    net_g = net_g.to(device)

    decoder = torch.load(args.decoder, map_location=device, weights_only=False)

    transcript = Path(args.transcript).read_text().strip()
    audio, sr = load_audio_mono(args.audio, TARGET_SAMPLE_RATE)
    spec = compute_spec(audio, sr)

    spec_chunks = chunk_spec(spec, MAX_SPEC_FRAMES)
    chunk_durations = [
        chunk.shape[-1] * HOP_LENGTH / TARGET_SAMPLE_RATE for chunk, _ in spec_chunks
    ]
    text_chunks = split_text_by_duration(transcript, chunk_durations)

    if len(spec_chunks) > 1:
        max_chunk_sec = MAX_SPEC_FRAMES * HOP_LENGTH / TARGET_SAMPLE_RATE
        print(f"Long audio detected ({len(spec_chunks)} chunks, max ~{max_chunk_sec:.1f}s each).")

    all_results = []
    for chunk_idx, ((chunk_spec, start_frame), text_chunk) in enumerate(
        zip(spec_chunks, text_chunks)
    ):
        if not text_chunk:
            text_chunk = transcript
        text_chunk = truncate_text_to_tokens(text_chunk, hps, MAX_TEXT_TOKENS)

        output = single_inference_from_spec(
            hps, net_g, chunk_spec, text_chunk, args.downsample_factor, decoder, device
        )

        disfluency_type_pred = output[:, :, 3:]
        type_log_probs = torch.log_softmax(disfluency_type_pred, dim=-1).squeeze(0)
        region_scores, region_labels = torch.max(type_log_probs, dim=-1)

        boundary_logits = output[:, :, :2].squeeze(0)
        frames_to_seconds = (MAX_SPEC_FRAMES * HOP_LENGTH) / TARGET_SAMPLE_RATE
        boundaries = boundary_logits * frames_to_seconds

        chunk_offset_sec = start_frame * HOP_LENGTH / TARGET_SAMPLE_RATE

        sorted_indices = torch.argsort(region_scores, descending=True)
        chunk_results = []
        for idx in sorted_indices[: args.top_k]:
            label_idx = region_labels[idx].item()
            start, end = boundaries[idx].tolist()
            chunk_results.append(
                {
                    "region_index": int(idx.item()),
                    "label": LABELS[label_idx],
                    "start_sec": float(start + chunk_offset_sec),
                    "end_sec": float(end + chunk_offset_sec),
                    "confidence": float(torch.exp(region_scores[idx]).item()),
                }
            )

        all_results.append(
            {
                "chunk_index": chunk_idx,
                "chunk_start_sec": float(chunk_offset_sec),
                "predictions": chunk_results,
            }
        )

    if len(all_results) == 1:
        first = all_results[0]["predictions"][0]
        output_obj = {"start": first["start_sec"], "end": first["end_sec"], "type": first["label"]}
    else:
        output_obj = {"chunks": all_results}

    print(json.dumps(output_obj, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(output_obj, indent=2))


if __name__ == "__main__":
    main()
