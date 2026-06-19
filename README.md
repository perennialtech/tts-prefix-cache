# tts-prefix-cache

Low-latency audio prefix splicing for speech and TTS pipelines.

`tts-prefix-cache` lets you play a cached synthesized prefix immediately while a full utterance is being generated in the background, then splice into the generated continuation once it is ready.

## When to use this

Use this package when your application often starts responses with repeated text, such as:

- “Sure —”
- “Let me check that.”
- “One moment while I look that up.”
- branded greetings or fixed assistant preambles

Instead of waiting for the whole utterance to synthesize, you can cache the prefix audio and start streaming it immediately.

## Install

From this repository:

```bash
pip install .
```

For local development with `uv`:

```bash
uv sync
```

## Audio contract

The synthesizer you provide should return mono, one-dimensional `numpy.float32` audio at the requested sample rate.

Audio samples are expected to be normal floating-point PCM-style samples. Helpers are available for WAV writing and PCM16LE conversion.

## Quick start

```python
import asyncio
import numpy as np

from tts_prefix_cache import BufferedWavSink, PrefixSpeaker, PrefixSpeakerConfig


class MySynthesizer:
    async def synthesize(self, text: str, *, sample_rate: int) -> np.ndarray:
        # Replace this with your TTS provider call.
        duration_seconds = max(0.2, len(text) * 0.03)
        samples = int(sample_rate * duration_seconds)

        # Demo placeholder audio.
        t = np.linspace(0, duration_seconds, samples, endpoint=False)
        return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


async def main() -> None:
    speaker = PrefixSpeaker(
        tts=MySynthesizer(),
        config=PrefixSpeakerConfig(sample_rate=24_000),
    )

    sink = BufferedWavSink(path="out/response.wav", sample_rate=24_000)

    result = await speaker.speak(
        prefix="Sure — ",
        rest="I can help with that.",
        sink=sink,
    )

    sink.save()
    print(result)


asyncio.run(main())
```

## Streaming audio chunks

Use `audio_stream` when you want NumPy audio chunks:

```python
async for chunk in speaker.audio_stream(
    prefix="Sure — ",
    rest="I can help with that.",
):
    send_audio_chunk(chunk)
```

Use `pcm16le_stream` when your transport expects raw little-endian 16-bit PCM bytes:

```python
async for data in speaker.pcm16le_stream(
    prefix="Sure — ",
    rest="I can help with that.",
):
    send_bytes(data)
```

## Stable cache identifiers

If your application has its own stable cache identifiers, pass a key:

```python
await speaker.speak(
    prefix="Sure — ",
    rest="I can help with that.",
    key=("voice-a", "sure"),
    sink=sink,
)
```

## Custom sinks

A sink only needs an async `write` method that accepts an audio chunk:

```python
class MySink:
    async def write(self, chunk: np.ndarray) -> None:
        ...
```

Then pass it to `speak`:

```python
await speaker.speak(
    prefix="Sure — ",
    rest="I can help with that.",
    sink=MySink(),
)
```

## Configuration

Runtime behavior is configured with `PrefixSpeakerConfig` and its nested splice configuration.

Prefer constructing those dataclasses directly in code so your editor and type checker can use the package’s type definitions as the source of truth:

```python
from tts_prefix_cache import PrefixSpeakerConfig, SpliceConfig

config = PrefixSpeakerConfig(
    sample_rate=24_000,
    splice=SpliceConfig(
        crossfade_ms=30.0,
    ),
)
```
