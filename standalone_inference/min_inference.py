import argparse
import json
from pathlib import Path

import torch

import util as U

LABELS = ["rep", "block", "missing", "replace", "prolong"]


def main():
    parser = argparse.ArgumentParser(description="Standalone YOLO-Stutter inference")
    parser.add_argument("--audio", required=True, help="Path to input wav")
    parser.add_argument("--transcript", required=True, help="Path to transcript text")
    parser.add_argument("--config", default=str(Path(__file__).with_name("ljs_base.json")))
    parser.add_argument("--model", default=str(Path(__file__).with_name("pretrained_ljs.pth")))
    parser.add_argument("--decoder", default=str(Path(__file__).with_name("decoder_tts_joint")))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--downsample-factor", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", help="Optional output JSON path")
    args = parser.parse_args()

    hps = U.get_hparams_from_file(args.config)
    net_g = U.SynthesizerTrn(
        len(U.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).eval()
    U.load_checkpoint(args.model, net_g, None)

    device = torch.device(args.device)
    net_g = net_g.to(device)

    U.register_decoder_aliases()
    decoder = torch.load(args.decoder, map_location=device, weights_only=False)

    transcript = Path(args.transcript).read_text().strip()
    audio, sr = U.load_audio_mono(args.audio, U.TARGET_SAMPLE_RATE)
    spec = U.compute_spec(audio, sr)

    spec_chunks = U.chunk_spec(spec, U.MAX_SPEC_FRAMES)
    chunk_durations = [
        chunk.shape[-1] * U.HOP_LENGTH / U.TARGET_SAMPLE_RATE for chunk, _ in spec_chunks
    ]
    text_chunks = U.split_text_by_duration(transcript, chunk_durations)

    if len(spec_chunks) > 1:
        max_chunk_sec = U.MAX_SPEC_FRAMES * U.HOP_LENGTH / U.TARGET_SAMPLE_RATE
        print(f"Long audio detected ({len(spec_chunks)} chunks, max ~{max_chunk_sec:.1f}s each).")

    all_results = []
    for chunk_idx, ((chunk_spec, start_frame), text_chunk) in enumerate(
        zip(spec_chunks, text_chunks)
    ):
        if not text_chunk:
            text_chunk = transcript
        text_chunk = U.truncate_text_to_tokens(text_chunk, hps, U.MAX_TEXT_TOKENS)

        output = U.single_inference_from_spec(
            hps, net_g, chunk_spec, text_chunk, args.downsample_factor, decoder, device
        )

        disfluency_type_pred = output[:, :, 3:]
        type_log_probs = torch.log_softmax(disfluency_type_pred, dim=-1).squeeze(0)
        region_scores, region_labels = torch.max(type_log_probs, dim=-1)

        boundary_logits = output[:, :, :2].squeeze(0)
        frames_to_seconds = (U.MAX_SPEC_FRAMES * U.HOP_LENGTH) / U.TARGET_SAMPLE_RATE
        boundaries = boundary_logits * frames_to_seconds

        chunk_offset_sec = start_frame * U.HOP_LENGTH / U.TARGET_SAMPLE_RATE

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
