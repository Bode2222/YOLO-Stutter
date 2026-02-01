import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import json
import re
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import utils.vits.commons as commons
import utils.vits.utils as utils
from utils.vits.models import SynthesizerTrn
from utils.vits.text.symbols import symbols
from utils.vits.text import text_to_sequence
from utils.vits.text import cleaners

from scipy.io.wavfile import write
from utils.vits.mel_processing import spectrogram_torch


import importlib
importlib.reload(utils)
from torch.nn.utils.rnn import pad_sequence
from utils.model_utils.conv1d_transformer import Conv1DTransformerDecoder

# from utils.model_utils.dataset_5 import Dataset
from tqdm import tqdm
import wave

import logging

matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

import numba.core.byteflow
numba_logger = logging.getLogger('numba.core.byteflow')
numba_logger.setLevel(logging.INFO)


import warnings
warnings.filterwarnings("ignore", category=UserWarning)



def safe_text_to_sequence(text, cleaner_names):
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    try_cleaners = cleaner_names
    try:
        for name in try_cleaners:
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
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def get_sample_rate_wave(audio_file_path):
    with wave.open(audio_file_path, 'rb') as wf:
        return wf.getframerate()


TARGET_SAMPLE_RATE = 22050
FILTER_LENGTH = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
MAX_SPEC_FRAMES = 1024
MAX_TEXT_TOKENS = 768

CONFIG_PATH = os.path.join(PROJECT_ROOT, "utils", "vits", "configs", "ljs_base.json")
MODEL_PATH = os.path.join(PROJECT_ROOT, "saved_models", "pretrained_ljs.pth")
DECODER_PATH = os.path.join(PROJECT_ROOT, "saved_models", "decoder_tts_joint")
DATA_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "inference"))

hps = utils.get_hparams_from_file(CONFIG_PATH)
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model)
_ = net_g.eval()

_ = utils.load_checkpoint(MODEL_PATH, net_g, None)


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
    return spec, audio_norm


def get_audio_a(filename):
    audio, sampling_rate = load_audio_mono(filename, TARGET_SAMPLE_RATE)
    spec, audio_norm = compute_spec(audio, sampling_rate)
    return spec, audio_norm


def get_labels(path):
    with open(path, "r") as f:
        labels = json.load(f)
    
    phonemes = labels[0]["phonemes"]
    text_path = path.replace("disfluent_labels", "gt_text")
    last = text_path.rfind('_')
    text_path = text_path[:last] + ".txt"
    with open(text_path, 'r') as file:
        text = file.read()

    # text = labels[0]["text"]

    for w in phonemes:
        w["start"] = int(w["start"] / 0.016)
        w["end"] = int(w["end"] / 0.016) 
    
    return phonemes, text

def get_audio(path):
    spec, wav = get_audio_a(path) #[spec is 1, d, t]

    return spec, wav 

def process_audio(spec, wav, _text):

    stn_tst = get_text(_text, hps)
    # stn_tst = get_text("knows both he them", hps)
    # stn_tst = get_text("I miss you", hps)
    x_tst = stn_tst.unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
    
    y = spec.unsqueeze(0)
    y_lengths = torch.LongTensor([y.shape[-1]])
    t = net_g(x_tst, x_tst_lengths, y, y_lengths) 
    # print(len(t[2]))
    o, l_length, (neg_cent, attn), ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x_tst, x_tst_lengths, y, y_lengths) #phoneme, mel_spec respectively

    # print("neg_cent shape: ", neg_cent.shape)

    neg_cent = neg_cent.squeeze(0)

    neg_cent = nn.functional.softmax(neg_cent, dim=1)

    return neg_cent


def get_soft_attention(hps, net_g, text, text_lengths, spec, spec_lengths):
    with torch.no_grad():
        o, l_length, (neg_cent, attn), ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(text, text_lengths, spec, spec_lengths)  # phoneme, mel_spec respectively

        neg_cent = nn.functional.softmax(neg_cent, dim=-1)

    return neg_cent


def single_inference_from_spec(hps, spec, ref_text, downsample_factor, decoder, device):
    text = get_text(ref_text, hps)
    text_length = torch.tensor(text.size(0))

    spec_length = torch.tensor(spec.shape[-1])

    text = text.unsqueeze(0)
    spec = spec.unsqueeze(0)
    text_length = text_length.unsqueeze(0)
    spec_length = spec_length.unsqueeze(0)

    num_regions = int(spec_length // downsample_factor)

    text = text.to(device)
    text_length = text_length.to(device)
    spec = spec.to(device)
    spec_length = spec_length.to(device)

    soft_attention = get_soft_attention(
        hps, net_g, text, text_length, spec, spec_length
    )

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

    output = decoder(new_soft_attention, mask.to(device))

    return output



def single_inference(hps, wav_path, ref_text, downsample_factor, decoder, device):
    ref_text = truncate_text_to_tokens(ref_text, hps, MAX_TEXT_TOKENS)
    text = get_text(ref_text, hps)
    text_length = torch.tensor(text.size(0))

    spec, wav = get_audio_a(wav_path)  # [1, d, t]
    if spec.shape[-1] > MAX_SPEC_FRAMES:
        spec = spec[:, :MAX_SPEC_FRAMES]
    spec_length = torch.tensor(spec.shape[-1])

    text = text.unsqueeze(0)
    spec = spec.unsqueeze(0)
    text_length = text_length.unsqueeze(0)
    spec_length = spec_length.unsqueeze(0)

    # print("text: ", text.shape) # [1, U]
    # print("spec: ", spec.shape) # [1, 513, T]
    # print(text_length)
    # print(spec_length)

    num_regions = int(spec_length // downsample_factor)

    text = text.to(device)
    text_length = text_length.to(device)
    spec = spec.to(device)
    spec_length = spec_length.to(device)

    soft_attention = get_soft_attention(
        hps, net_g, text, text_length, spec, spec_length
    )

    orig_text_dim_shape = soft_attention.shape[-1]
    
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

    output = decoder(new_soft_attention, mask.to(device))

    return output


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


def resolve_input_paths(data_dir):
    data_dir = Path(data_dir)

    audio_path = None
    for name in ("raw_22050_mono.wav", "raw.wav"):
        candidate = data_dir / name
        if candidate.exists():
            audio_path = candidate
            break
    if audio_path is None:
        wavs = sorted(data_dir.glob("*.wav"))
        if wavs:
            audio_path = wavs[0]

    transcript_path = None
    for name in ("raw_transcript.txt", "transcript.txt"):
        candidate = data_dir / name
        if candidate.exists():
            transcript_path = candidate
            break
    if transcript_path is None:
        txts = sorted(data_dir.glob("*.txt"))
        if txts:
            transcript_path = txts[0]

    if audio_path is None:
        raise FileNotFoundError(f"No .wav file found under {data_dir}")
    if transcript_path is None:
        raise FileNotFoundError(f"No transcript .txt file found under {data_dir}")

    return str(audio_path), str(transcript_path)


if __name__ == "__main__":

    device_name = os.environ.get("YOLOSTUTTER_DEVICE")
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cpu")

    text_channels = 768
    kernel_size= 3
    kernel_stride = 1
    num_blocks = 4
    num_classes = 5   ## change
    downsample_factor = 16  #8
    n_heads = 8
    n_layers = 8


    decoder = torch.load(DECODER_PATH, map_location=device, weights_only=False)
    
    labels = ["rep", "block", "missing", "replace", "prolong"]

    net_g = net_g.to(device)

    wav_path, transcript_path = resolve_input_paths(DATA_DIR)
    with open(transcript_path, "r") as f:
        ref_text = f.read().strip()

    audio, sampling_rate = load_audio_mono(wav_path, TARGET_SAMPLE_RATE)
    spec, _ = compute_spec(audio, sampling_rate)

    spec_chunks = chunk_spec(spec, MAX_SPEC_FRAMES)
    chunk_durations = [
        chunk.shape[-1] * HOP_LENGTH / TARGET_SAMPLE_RATE for chunk, _ in spec_chunks
    ]
    text_chunks = split_text_by_duration(ref_text, chunk_durations)

    if len(spec_chunks) > 1:
        max_chunk_sec = MAX_SPEC_FRAMES * HOP_LENGTH / TARGET_SAMPLE_RATE
        print(
            f"Long audio detected ({len(spec_chunks)} chunks, max ~{max_chunk_sec:.1f}s each)."
        )

    all_results = []

    for chunk_idx, ((chunk_spec, start_frame), text_chunk) in enumerate(
        zip(spec_chunks, text_chunks)
    ):
        if not text_chunk:
            text_chunk = ref_text
        text_chunk = truncate_text_to_tokens(text_chunk, hps, MAX_TEXT_TOKENS)

        output = single_inference_from_spec(
            hps, chunk_spec, text_chunk, downsample_factor, decoder, device
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
        for idx in sorted_indices[:5]:
            label_idx = region_labels[idx].item()
            start, end = boundaries[idx].tolist()
            chunk_results.append(
                {
                    "region_index": int(idx.item()),
                    "label": labels[label_idx],
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
        print(
            {
                "start": first["start_sec"],
                "end": first["end_sec"],
                "type": first["label"],
            }
        )
    else:
        print(json.dumps({"chunks": all_results}, indent=2))
