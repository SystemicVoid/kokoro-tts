#!/usr/bin/env python3
"""Local TTS CLI — kokoro-onnx → PCM → pw-cat (PipeWire)."""

import argparse
import asyncio
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MODEL_DIR = Path.home() / ".local" / "share" / "kokoro-tts"
BASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
MODEL_FILE = "kokoro-v1.0.int8.onnx"
VOICES_FILE = "voices-v1.0.bin"


def strip_markdown(text: str) -> str:
    """Remove markdown formatting, keeping readable prose."""
    text = re.sub(r"```[\s\S]*?```", "", text)  # fenced code blocks
    text = re.sub(r"`([^`]+)`", r"\1", text)  # inline code
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)  # images
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)  # headings
    text = re.sub(r"^[>\-*_~]+\s?", "", text, flags=re.MULTILINE)  # block markers
    text = re.sub(r"[*_~]{1,3}", "", text)  # bold/italic/strikethrough
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse blank lines
    return text.strip()


def get_text(arg: str | None) -> str:
    """Resolve input text from argument, file, or stdin."""
    if arg:
        p = Path(arg)
        if p.is_file():
            text = p.read_text(encoding="utf-8", errors="replace")
            if p.suffix.lower() == ".md":
                text = strip_markdown(text)
            return text
        return arg
    if not sys.stdin.isatty():
        return sys.stdin.read()
    print("Usage: tts [--voice V] [--speed S] [--lang L] <text or file>", file=sys.stderr)
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
            print(f"Downloading {fname}…", file=sys.stderr)
            tmp = dest.with_suffix(".tmp")
            urllib.request.urlretrieve(url, tmp)  # noqa: S310
            tmp.rename(dest)
            print(f"  → {dest}", file=sys.stderr)
    return model_path, voices_path


def find_player() -> str:
    """Find pw-cat or pacat for audio playback."""
    for cmd in ("pw-cat", "pacat"):
        if shutil.which(cmd):
            return cmd
    print("Error: neither pw-cat nor pacat found. Install PipeWire or PulseAudio.", file=sys.stderr)
    sys.exit(1)


def chunk_text(text: str) -> list[str]:
    """Split on paragraph boundaries for incremental phonemization.

    create_stream() phonemizes the entire input before yielding audio.
    Feeding paragraphs one at a time lets the first chunk start playing
    while later paragraphs are still being phonemized.
    """
    chunks = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return chunks or [text]


def _log(msg: str) -> None:
    print(f"[tts] {msg}", file=sys.stderr)


async def speak(
    text: str, voice: str, speed: float, lang: str,
    threads: int, no_spin: bool, stats: bool,
) -> None:
    """Stream TTS audio paragraph-by-paragraph to PipeWire."""
    import numpy as np
    import onnxruntime as ort
    from kokoro_onnx import Kokoro

    model_path, voices_path = ensure_models()

    t0 = time.perf_counter()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if no_spin:
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")

    session = ort.InferenceSession(
        str(model_path), opts, providers=["CPUExecutionProvider"]
    )
    kokoro = Kokoro.from_session(session, str(voices_path))

    if stats:
        _log(f"model: {time.perf_counter() - t0:.2f}s ({threads} threads)")

    player_cmd = find_player()
    proc = None
    chunks = chunk_text(text)

    if stats:
        _log(f"{len(chunks)} paragraph(s), {len(text)} chars total")

    total_audio_dur = 0.0
    t_wall_start = time.perf_counter()

    try:
        for i, chunk in enumerate(chunks):
            t_para_start = time.perf_counter()
            n_sents = 0
            para_audio_dur = 0.0
            t_first_audio = None

            stream = kokoro.create_stream(chunk, voice=voice, speed=speed, lang=lang)
            async for samples, sample_rate in stream:
                if t_first_audio is None:
                    t_first_audio = time.perf_counter() - t_para_start
                n_sents += 1
                para_audio_dur += len(samples) / sample_rate

                if proc is None:
                    proc = subprocess.Popen(
                        [
                            player_cmd, "--playback", "--raw",
                            "--rate", str(sample_rate),
                            "--channels", "1",
                            "--format", "s16",
                            "-",
                        ],
                        stdin=subprocess.PIPE,
                    )
                pcm = np.clip(samples * 32767, -32768, 32767).astype(np.int16).tobytes()
                proc.stdin.write(pcm)

            total_audio_dur += para_audio_dur
            if stats and t_first_audio is not None:
                buffer_s = total_audio_dur - (time.perf_counter() - t_wall_start)
                preview = chunk[:60].replace("\n", " ")
                if len(chunk) > 60:
                    preview += "…"
                _log(
                    f"¶{i + 1} \"{preview}\"\n"
                    f"      {len(chunk):>4}ch {n_sents}sn | "
                    f"wait {t_first_audio:.2f}s → {para_audio_dur:.1f}s audio | "
                    f"buffer {buffer_s:+.1f}s"
                )
    except BrokenPipeError:
        pass
    finally:
        if proc is not None:
            proc.stdin.close()
            proc.wait()

    if stats:
        wall = time.perf_counter() - t_wall_start
        _log(f"done: {total_audio_dur:.1f}s audio, {wall:.1f}s wall, "
             f"{total_audio_dur / wall:.1f}x realtime")


def main() -> None:
    default_threads = int(os.environ.get("TTS_THREADS", "8"))
    default_no_spin = os.environ.get("TTS_NO_SPIN", "") == "1"

    parser = argparse.ArgumentParser(
        description="Local TTS via kokoro-onnx → PipeWire"
    )
    parser.add_argument("text", nargs="?", help="Text string or path to .txt/.md file")
    parser.add_argument("--voice", default="af_heart", help="Voice name (default: af_heart)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (default: 1.0)")
    parser.add_argument("--lang", default="en-us", help="Language code (default: en-us)")
    parser.add_argument(
        "--threads", type=int, default=default_threads,
        help=f"ONNX intra-op threads (default: {default_threads}, env: TTS_THREADS)",
    )
    parser.add_argument(
        "--no-spin", action="store_true", default=default_no_spin,
        help="Disable ONNX thread spinning (lower idle CPU, env: TTS_NO_SPIN=1)",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print timing stats to stderr",
    )
    args = parser.parse_args()

    text = get_text(args.text)
    if not text.strip():
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    try:
        asyncio.run(speak(
            text, voice=args.voice, speed=args.speed, lang=args.lang,
            threads=args.threads, no_spin=args.no_spin, stats=args.stats,
        ))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
