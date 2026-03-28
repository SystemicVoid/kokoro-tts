#!/usr/bin/env python3
"""Local TTS CLI - kokoro-onnx to PCM to system audio playback."""

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MODEL_DIR = Path.home() / ".local" / "share" / "kokoro-tts"
BASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
MODEL_FILE = "kokoro-v1.0.int8.onnx"
VOICES_FILE = "voices-v1.0.bin"

ChunkMode = Literal["document", "sentences", "paragraphs"]
SegmentKind = Literal["prose", "heading", "list", "table"]
INLINE_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
INLINE_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
RAW_URL_RE = re.compile(r"https?://\S+")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")
DEFAULT_SENTENCE_TARGET_MIN = 120
DEFAULT_SENTENCE_TARGET_MAX = 250


@dataclass(slots=True)
class Segment:
    kind: SegmentKind
    text: str


def _clean_inline_text(text: str) -> str:
    text = INLINE_IMAGE_RE.sub(r"\1", text)
    text = INLINE_LINK_RE.sub(r"\1", text)
    text = RAW_URL_RE.sub("", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"[*_~]{1,3}", "", text)
    text = text.replace("|", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,;-")


def _parse_segments(text: str) -> list[Segment]:
    text = re.sub(r"```[\s\S]*?```", "", text)
    segments: list[Segment] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        paragraph = _clean_inline_text(" ".join(paragraph_lines))
        paragraph_lines.clear()
        if paragraph:
            segments.append(Segment("prose", paragraph))

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            continue

        stripped = re.sub(r"^\s*>\s?", "", stripped)

        if TABLE_SEPARATOR_RE.match(stripped):
            flush_paragraph()
            continue

        heading_match = re.match(r"^\s*#{1,6}\s+(.*)$", stripped)
        if heading_match:
            flush_paragraph()
            heading = _clean_inline_text(heading_match.group(1))
            if heading:
                segments.append(Segment("heading", heading))
            continue

        bullet_match = re.match(r"^\s*(?:[-*+]|\d+\.)\s+(.*)$", stripped)
        if bullet_match:
            flush_paragraph()
            item = _clean_inline_text(bullet_match.group(1))
            if item:
                segments.append(Segment("list", item))
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph()
            cells = [_clean_inline_text(cell) for cell in stripped.strip("|").split("|")]
            cells = [cell for cell in cells if cell]
            if cells:
                segments.append(Segment("table", ", ".join(cells)))
            continue

        paragraph_lines.append(stripped)

    flush_paragraph()
    return segments


def _merge_short_headings(segments: list[Segment]) -> list[Segment]:
    merged: list[Segment] = []
    pending_heading: str | None = None

    for segment in segments:
        if segment.kind == "heading" and len(segment.text) <= 30:
            pending_heading = segment.text if pending_heading is None else f"{pending_heading}. {segment.text}"
            continue

        if pending_heading and segment.kind != "heading":
            merged.append(Segment(segment.kind, f"{pending_heading}. {segment.text}"))
            pending_heading = None
            continue

        if pending_heading:
            merged.append(Segment("prose", pending_heading))
            pending_heading = None

        merged.append(segment)

    if pending_heading:
        merged.append(Segment("prose", pending_heading))

    return merged


def normalized_segments(text: str) -> list[Segment]:
    return _merge_short_headings(_parse_segments(text))


def normalize_for_tts(text: str) -> str:
    return "\n\n".join(segment.text for segment in normalized_segments(text))


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(text) if part.strip()]
    return parts or [text]


def estimate_sentence_count(text: str) -> int:
    return max(1, len(split_sentences(text)))


def _build_sentence_chunks(
    segments: list[Segment],
    target_min: int = DEFAULT_SENTENCE_TARGET_MIN,
    target_max: int = DEFAULT_SENTENCE_TARGET_MAX,
) -> list[str]:
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    def flush_current() -> None:
        nonlocal current_parts, current_len
        if not current_parts:
            return
        chunks.append(" ".join(current_parts))
        current_parts = []
        current_len = 0

    def push_sentence(sentence: str) -> None:
        nonlocal current_len, current_parts
        sentence = sentence.strip()
        if not sentence:
            return

        added_len = len(sentence) if not current_parts else len(sentence) + 1
        would_exceed = current_parts and current_len + added_len > target_max
        if would_exceed and current_len >= target_min:
            flush_current()
            added_len = len(sentence)

        if current_parts:
            current_parts.append(sentence)
            current_len += added_len
        else:
            current_parts = [sentence]
            current_len = len(sentence)

    for segment in segments:
        if segment.kind in {"list", "table"}:
            flush_current()
            chunks.append(segment.text)
            continue

        for sentence in split_sentences(segment.text):
            push_sentence(sentence)

    flush_current()
    return chunks


def plan_chunks(text: str, mode: ChunkMode) -> list[str]:
    segments = normalized_segments(text)
    if not segments:
        return []
    if mode == "document":
        return ["\n\n".join(segment.text for segment in segments)]
    if mode == "paragraphs":
        return [segment.text for segment in segments]
    if mode == "sentences":
        return _build_sentence_chunks(segments)
    raise ValueError(f"unsupported chunk mode: {mode}")


def get_text(arg: str | None) -> str:
    """Resolve input text from argument, file, or stdin."""
    if arg:
        p = Path(arg)
        if p.is_file():
            return p.read_text(encoding="utf-8", errors="replace")
        return arg
    if not sys.stdin.isatty():
        return sys.stdin.read()
    print(
        "Usage: tts [--voice V] [--speed S] [--lang L] [--chunk-mode M] <text or file>",
        file=sys.stderr,
    )
    print("       echo 'text' | tts", file=sys.stderr)
    sys.exit(1)


def ensure_models() -> tuple[Path, Path]:
    """Download model files on first run, return (model_path, voices_path)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / MODEL_FILE
    voices_path = MODEL_DIR / VOICES_FILE
    for fname, dest in [(MODEL_FILE, model_path), (VOICES_FILE, voices_path)]:
        if not dest.exists():
            url = f"{BASE_URL}/{fname}"
            print(f"Downloading {fname}...", file=sys.stderr)
            tmp = dest.with_suffix(".tmp")
            urllib.request.urlretrieve(url, tmp)  # noqa: S310
            tmp.rename(dest)
            print(f"  -> {dest}", file=sys.stderr)
    return model_path, voices_path


def build_kokoro(threads: int, no_spin: bool):
    import onnxruntime as ort
    from kokoro_onnx import Kokoro

    model_path, voices_path = ensure_models()

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if no_spin:
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")

    session = ort.InferenceSession(
        str(model_path), opts, providers=["CPUExecutionProvider"]
    )
    return Kokoro.from_session(session, str(voices_path))


def find_player() -> str:
    """Find pw-cat or pacat for audio playback."""
    for cmd in ("pw-cat", "pacat"):
        if shutil.which(cmd):
            return cmd
    print("Error: neither pw-cat nor pacat found. Install PipeWire or PulseAudio.", file=sys.stderr)
    sys.exit(1)


def build_player_command(player_cmd: str, sample_rate: int) -> list[str]:
    """Build the playback command for the selected audio backend."""
    base_args = ["--raw", "--rate", str(sample_rate), "--channels", "1"]
    if player_cmd == "pw-cat":
        return [player_cmd, "--playback", *base_args, "--format", "s16", "-"]
    return [player_cmd, "--playback", *base_args, "--format", "s16le"]


def _log(msg: str) -> None:
    print(f"[tts] {msg}", file=sys.stderr)


def write_stats_json(path: str | None, metrics: dict) -> None:
    if not path:
        return
    stats_path = Path(path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


async def speak(
    text: str,
    voice: str,
    speed: float,
    lang: str,
    threads: int,
    no_spin: bool,
    stats: bool,
    dry_run: bool,
    chunk_mode: ChunkMode,
    stats_json: str | None = None,
) -> dict:
    """Stream TTS audio using the selected chunking strategy."""
    model_start = time.perf_counter()
    kokoro = build_kokoro(threads, no_spin)
    model_load_s = time.perf_counter() - model_start

    normalize_start = time.perf_counter()
    normalized_text = normalize_for_tts(text)
    normalize_s = time.perf_counter() - normalize_start

    plan_start = time.perf_counter()
    chunks = plan_chunks(text, chunk_mode)
    plan_s = time.perf_counter() - plan_start

    if stats:
        _log(f"model: {model_load_s:.2f}s ({threads} threads)")
        _log(
            f"normalize: {normalize_s:.3f}s | plan: {plan_s:.3f}s | "
            f"{len(chunks)} chunk(s) via {chunk_mode}"
        )

    player_cmd = find_player() if not dry_run else None
    proc = None
    np = None
    first_output_at: float | None = None
    total_audio_dur = 0.0
    wall_start = time.perf_counter()
    chunk_metrics: list[dict] = []

    try:
        for i, chunk in enumerate(chunks):
            chunk_start = time.perf_counter()
            audio_dur = 0.0
            first_chunk_output_at: float | None = None
            chunk_last_output_at: float | None = None
            outputs = 0

            stream = kokoro.create_stream(chunk, voice=voice, speed=speed, lang=lang)
            async for samples, sample_rate in stream:
                now = time.perf_counter()
                if first_output_at is None:
                    first_output_at = now
                if first_chunk_output_at is None:
                    first_chunk_output_at = now

                outputs += 1
                audio_dur += len(samples) / sample_rate

                if not dry_run:
                    if np is None:
                        import numpy as np_module

                        np = np_module
                    if proc is None:
                        proc = subprocess.Popen(
                            build_player_command(player_cmd, sample_rate),
                            stdin=subprocess.PIPE,
                        )
                    pcm = np.clip(samples * 32767, -32768, 32767).astype(np.int16).tobytes()
                    proc.stdin.write(pcm)

                chunk_last_output_at = time.perf_counter()

            wait_s = None if first_chunk_output_at is None else first_chunk_output_at - chunk_start
            gap_s = None
            if chunk_metrics and chunk_metrics[-1]["last_output_at_s"] is not None and first_chunk_output_at is not None:
                gap_s = first_chunk_output_at - chunk_metrics[-1]["last_output_at_s"]

            total_audio_dur += audio_dur
            preview = chunk[:60].replace("\n", " ")
            if len(chunk) > 60:
                preview += "..."

            chunk_metric = {
                "index": i + 1,
                "chars": len(chunk),
                "sentences_estimate": estimate_sentence_count(chunk),
                "outputs": outputs,
                "wait_s": wait_s,
                "audio_s": audio_dur,
                "inter_chunk_gap_s": gap_s,
                "preview": preview,
                "last_output_at_s": chunk_last_output_at,
            }
            chunk_metrics.append(chunk_metric)

            if stats and wait_s is not None:
                gap_label = "n/a" if gap_s is None else f"{gap_s:+.2f}s"
                _log(
                    f"#{i + 1} \"{preview}\"\n"
                    f"      {len(chunk):>4}ch {chunk_metric['sentences_estimate']}sn | "
                    f"wait {wait_s:.2f}s -> {audio_dur:.1f}s audio | gap {gap_label}"
                )
    except BrokenPipeError:
        pass
    finally:
        if proc is not None:
            proc.stdin.close()
            proc.wait()

    wall_s = time.perf_counter() - wall_start
    first_audio_s = None if first_output_at is None else first_output_at - wall_start
    realtime_factor = total_audio_dur / wall_s if wall_s > 0 else 0.0
    metrics = {
        "chunk_mode": chunk_mode,
        "input_chars": len(text),
        "normalized_chars": len(normalized_text),
        "chunk_count": len(chunks),
        "model_load_s": model_load_s,
        "normalize_s": normalize_s,
        "plan_s": plan_s,
        "time_to_first_audio_s": first_audio_s,
        "total_audio_s": total_audio_dur,
        "wall_s": wall_s,
        "realtime_factor": realtime_factor,
        "chunks": [
            {key: value for key, value in chunk.items() if key != "last_output_at_s"}
            for chunk in chunk_metrics
        ],
    }
    write_stats_json(stats_json, metrics)

    if stats:
        first_audio_label = "n/a" if first_audio_s is None else f"{first_audio_s:.2f}s"
        _log(
            f"done: first audio {first_audio_label}, "
            f"{total_audio_dur:.1f}s audio, {wall_s:.1f}s wall, {realtime_factor:.1f}x realtime"
        )

    return metrics


def main() -> None:
    default_threads = int(os.environ.get("TTS_THREADS", "8"))
    default_no_spin = os.environ.get("TTS_NO_SPIN", "") == "1"

    parser = argparse.ArgumentParser(
        description="Local TTS via kokoro-onnx with system audio playback"
    )
    parser.add_argument("text", nargs="?", help="Text string or path to .txt/.md file")
    parser.add_argument("--voice", default="af_heart", help="Voice name (default: af_heart)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (default: 1.0)")
    parser.add_argument("--lang", default="en-us", help="Language code (default: en-us)")
    parser.add_argument(
        "--threads",
        type=int,
        default=default_threads,
        help=f"ONNX intra-op threads (default: {default_threads}, env: TTS_THREADS)",
    )
    parser.add_argument(
        "--no-spin",
        action="store_true",
        default=default_no_spin,
        help="Disable ONNX thread spinning (lower idle CPU, env: TTS_NO_SPIN=1)",
    )
    parser.add_argument(
        "--chunk-mode",
        choices=("document", "sentences", "paragraphs"),
        default="document",
        help="Chunking strategy: full document, adaptive sentences, or paragraphs",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print timing stats to stderr",
    )
    parser.add_argument(
        "--stats-json",
        help="Write run metrics as JSON to the given path",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run synthesis without playing audio (for benchmarking with --stats)",
    )
    args = parser.parse_args()

    text = get_text(args.text)
    if not text.strip():
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    try:
        asyncio.run(
            speak(
                text=text,
                voice=args.voice,
                speed=args.speed,
                lang=args.lang,
                threads=args.threads,
                no_spin=args.no_spin,
                stats=args.stats,
                dry_run=args.dry_run,
                chunk_mode=args.chunk_mode,
                stats_json=args.stats_json,
            )
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
