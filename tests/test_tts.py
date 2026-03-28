import asyncio
import json

import tts


class FakeKokoro:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create_stream(self, text: str, voice: str, speed: float, lang: str):
        self.calls.append(
            {
                "text": text,
                "voice": voice,
                "speed": speed,
                "lang": lang,
            }
        )

        async def generator():
            yield [0.0] * 2400, 24000
            yield [0.0] * 1200, 24000

        return generator()


def test_normalize_for_tts_attaches_headings_and_cleans_markdown() -> None:
    text = """
# Intro
Visit [the docs](https://example.com/docs) and https://example.com/raw

- First bullet
- Second bullet

| Name | Value |
| ---- | ----- |
| CPU | Fast |

```python
print("drop me")
```
"""

    normalized = tts.normalize_for_tts(text)

    assert "Intro. Visit the docs and" in normalized
    assert "https://" not in normalized
    assert "drop me" not in normalized
    assert "First bullet" in normalized
    assert "Second bullet" in normalized
    assert "Name, Value" in normalized
    assert "CPU, Fast" in normalized


def test_plan_chunks_document_returns_single_normalized_chunk() -> None:
    text = "# Intro\nHello world.\n\nSecond paragraph."

    chunks = tts.plan_chunks(text, "document")

    assert chunks == ["Intro. Hello world.\n\nSecond paragraph."]


def test_plan_chunks_sentences_isolates_lists_and_tables() -> None:
    text = """
# Intro
Alpha sentence. Beta sentence. Gamma sentence.

- Bullet item

Delta sentence. Epsilon sentence.

| CPU | Fast |
"""

    chunks = tts.plan_chunks(text, "sentences")

    assert "Bullet item" in chunks
    assert "CPU, Fast" in chunks
    assert chunks[0].startswith("Intro. Alpha sentence.")
    assert chunks[1] == "Bullet item"
    assert chunks[-1] == "CPU, Fast"


def test_speak_document_mode_uses_single_stream_call_and_writes_stats_json(
    monkeypatch, tmp_path
) -> None:
    fake = FakeKokoro()
    stats_path = tmp_path / "stats.json"

    monkeypatch.setattr(tts, "build_kokoro", lambda threads, no_spin: fake)

    metrics = asyncio.run(
        tts.speak(
            text="# Intro\nHello world.\n\nSecond paragraph.",
            voice="af_heart",
            speed=1.0,
            lang="en-us",
            threads=4,
            no_spin=False,
            stats=False,
            dry_run=True,
            chunk_mode="document",
            stats_json=str(stats_path),
        )
    )

    payload = json.loads(stats_path.read_text(encoding="utf-8"))

    assert len(fake.calls) == 1
    assert fake.calls[0]["text"] == "Intro. Hello world.\n\nSecond paragraph."
    assert metrics["chunk_count"] == 1
    assert payload["chunk_mode"] == "document"
    assert payload["chunk_count"] == 1
    assert payload["chunks"][0]["chars"] == len(fake.calls[0]["text"])


def test_speak_paragraph_mode_uses_multiple_stream_calls(monkeypatch) -> None:
    fake = FakeKokoro()

    monkeypatch.setattr(tts, "build_kokoro", lambda threads, no_spin: fake)

    metrics = asyncio.run(
        tts.speak(
            text="# Intro\nHello world.\n\nSecond paragraph.",
            voice="af_heart",
            speed=1.0,
            lang="en-us",
            threads=4,
            no_spin=False,
            stats=False,
            dry_run=True,
            chunk_mode="paragraphs",
        )
    )

    assert [call["text"] for call in fake.calls] == [
        "Intro. Hello world.",
        "Second paragraph.",
    ]
    assert metrics["chunk_count"] == 2


def test_parser_defaults_to_sentences_mode() -> None:
    parser = tts.build_parser()

    args = parser.parse_args(["hello world"])

    assert args.chunk_mode == "sentences"
