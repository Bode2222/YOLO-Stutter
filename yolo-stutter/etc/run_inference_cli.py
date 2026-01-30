import argparse
import json
import os
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ORIG_CWD = Path.cwd()
os.chdir(SCRIPT_DIR)
import inference  # noqa: E402
os.chdir(ORIG_CWD)

LABELS = ["rep", "block", "missing", "replace", "prolong"]
DEFAULT_DECODER = SCRIPT_DIR.parent / "saved_models" / "decoder_tts_joint"

def choose_device(preferred: str) -> torch.device:
    preferred = preferred.lower()
    if preferred.startswith("cuda") and torch.cuda.is_available():
        return torch.device(preferred)
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_text(args: argparse.Namespace) -> str:
    if args.text_string:
        return args.text_string.strip()
    if args.text:
        text_path = Path(args.text)
        return text_path.read_text().strip()
    raise ValueError("Either --text or --text-string must be provided")

def run_inference(audio_path: str, transcript: str, decoder_path: Path, device: torch.device, downsample_factor: int, top_k: int):
    inference.net_g = inference.net_g.to(device)
    # The checkpoint stores a full module object, so load with weights_only=False.
    decoder = torch.load(decoder_path, map_location=device, weights_only=False)
    output = inference.single_inference(
        inference.hps,
        audio_path,
        transcript,
        downsample_factor,
        decoder,
        device,
    )

    disfluency_type_pred = output[:, :, 3:]
    type_log_probs = torch.log_softmax(disfluency_type_pred, dim=-1).squeeze(0)
    region_scores, region_labels = torch.max(type_log_probs, dim=-1)

    boundary_logits = output[:, :, :2].squeeze(0)
    # convert to seconds (see original notebook)
    frames_to_seconds = (1024 * 256) / 22050
    boundaries = boundary_logits * frames_to_seconds

    sorted_indices = torch.argsort(region_scores, descending=True)
    results = []
    for idx in sorted_indices[:top_k]:
        label_idx = region_labels[idx].item()
        label = LABELS[label_idx]
        start, end = boundaries[idx].tolist()
        results.append(
            {
                "region_index": int(idx.item()),
                "label": label,
                "start_sec": float(start),
                "end_sec": float(end),
                "confidence": float(torch.exp(region_scores[idx]).item()),
            }
        )
    return results

def main():
    parser = argparse.ArgumentParser(description="YOLO-Stutter CLI inference helper")
    parser.add_argument("--audio", required=True, help="Path to the input wav file")
    parser.add_argument("--text", help="Path to transcript text file")
    parser.add_argument("--text-string", help="Transcript text provided directly via CLI")
    parser.add_argument("--decoder", default=str(DEFAULT_DECODER), help="Path to decoder checkpoint")
    parser.add_argument("--device", default="cuda", help="Torch device spec (e.g., cuda, cuda:0, cpu)")
    parser.add_argument("--downsample-factor", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", help="Optional path to write predictions as JSON")
    args = parser.parse_args()

    transcript = load_text(args)
    device = choose_device(args.device)

    results = run_inference(
        audio_path=args.audio,
        transcript=transcript,
        decoder_path=Path(args.decoder),
        device=device,
        downsample_factor=args.downsample_factor,
        top_k=args.top_k,
    )

    print(json.dumps({"predictions": results}, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"predictions": results}, indent=2))

if __name__ == "__main__":
    main()
