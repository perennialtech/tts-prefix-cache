# tts-prefix-cache

Low-latency audio prefix splicing for speech and TTS pipelines.

`tts-prefix-cache` lets you play a cached synthesized prefix immediately while a full utterance is being generated in the background, then splice into the generated continuation once it is ready.

This library intentionally does not expose a public prefix warming API because we do not need it internally.

## When to use this

Use this package when your application often starts responses with repeated text, such as:

- "Sure,"
- "Let me check that."
- "One moment while I look that up."
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

## Quick start: offline rendering

Use `render()` when you want a complete audio array or WAV file. Offline rendering does not sleep, does not simulate playback, and does not inject latency silence.

```python
import asyncio
import numpy as np

from tts_prefix_cache import PrefixSpeaker, PrefixSpeakerConfig, write_wav


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

    audio, result = await speaker.render(
        prefix="Sure, ",
        continuation="I can help with that.",
    )

    write_wav("out/response.wav", audio, sample_rate=24_000)
    print(result)


asyncio.run(main())
```

## Live playback

Use `speak()` when you are feeding a live sink. Live playback makes timing decisions: when to release the held prefix tail, when to splice, and when to inject same-stream silence if full synthesis is late.

```python
from tts_prefix_cache import PrefixSpeaker, PrefixSpeakerConfig

config = PrefixSpeakerConfig(
    sample_rate=24_000,
    output_lead_ms=20.0,
)

speaker = PrefixSpeaker(tts=MySynthesizer(), config=config)
sink = MyLiveAudioSink()

result = await speaker.speak(
    prefix="Sure, ",
    continuation="I can help with that.",
    sink=sink,
)
```

## Live playback timing

The library uses a local buffered playout timeline.

It writes already-known audio as fast as the sink accepts it, while tracking how much audio has been queued for downstream playback. It does not inject silence merely because Python finished writing cached prefix bytes quickly. It waits until the queued known audio is close to running out, then decides whether to splice, release the held prefix tail, or add same-stream silence.

This is suitable for:

- `audio_stream()`
- simple bounded queues
- WebSocket senders without direct playout feedback
- in-memory sinks
- sinks that copy, enqueue, or send data

The sink should not sleep to simulate playback. Downstream audio infrastructure should handle actual playback pacing.

## Streaming audio chunks

Use `audio_stream()` when you want NumPy audio chunks:

```python
stream = speaker.audio_stream(
    prefix="Sure, ",
    continuation="I can help with that.",
)

async for chunk in stream:
    send_audio_chunk(chunk)

result = await stream.result()
```

Use `pcm16le_stream()` when your transport expects raw little-endian 16-bit PCM bytes:

```python
stream = speaker.pcm16le_stream(
    prefix="Sure, ",
    continuation="I can help with that.",
)

async for data in stream:
    send_bytes(data)

result = await stream.result()
```

Drain the stream before awaiting `result()`. If you stop consuming early, call `await stream.aclose()` or use `async with` so the background synthesis task is cancelled cleanly:

```python
async with speaker.audio_stream(
    prefix="Sure, ",
    continuation="I can help with that.",
) as stream:
    async for chunk in stream:
        send_audio_chunk(chunk)

    result = await stream.result()
```

## Stable cache identifiers and correctness

By default, the in-memory cache key is scoped to the `PrefixSpeaker` instance and includes the sample rate, splice configuration, and exact prefix text. This prevents accidental reuse across different speakers or splice settings when a cache object is shared.

If your application needs stable identifiers across speakers, restarts, workers, or processes, pass `cache_key` explicitly:

```python
await speaker.speak(
    prefix="Sure, ",
    continuation="I can help with that.",
    cache_key=(
        "provider-a",
        "model-x",
        "voice-a",
        24_000,
        "default-style",
        speaker.config.splice,
        "Sure, ",
    ),
    sink=sink,
)
```

The cache key must identify every input that can change either the prefix waveform or the prepared prefix data. At minimum, include:

- TTS provider
- model
- voice
- sample rate
- speaking rate, style, emotion, language, or prompt settings
- the exact prefix text, including whitespace and punctuation
- the splice configuration
- any other synthesis option that can alter timing, prosody, loudness, or audio format

Do not reuse a cache key across different voices, models, sample rates, style settings, or splice configurations. A wrong cache key can make the library splice prefix audio from one synthesis configuration into full audio from another, producing bad alignment, audible jumps, or incorrect playback.

## Standalone prefix prosody

The cached prefix is synthesized by itself, while the full utterance is synthesized as `prefix + continuation`.

Some TTS systems render a standalone phrase differently from the same phrase at the start of a longer sentence. The library aligns and crossfades the audio, but best results come from prefixes whose standalone rendering is close to their rendering inside the full utterance.

Stable short prefixes such as "Sure, " or "One moment, " usually work better than prefixes whose prosody strongly depends on the following words.

## Custom sinks

A sink only needs an async `write` method that accepts an audio chunk:

```python
class MySink:
    async def write(self, chunk: np.ndarray) -> None:
        ...
```

Then pass it to `speak()`:

```python
await speaker.speak(
    prefix="Sure, ",
    continuation="I can help with that.",
    sink=MySink(),
)
```

The sink may copy, enqueue, or send data. Downstream playback infrastructure should handle real playback pacing.

## Configuration

Runtime behavior is configured with `PrefixSpeakerConfig` and its nested splice configuration.

Prefer constructing those dataclasses directly in code so your editor and type checker can use the package's type definitions as the source of truth:

```python
from tts_prefix_cache import PrefixSpeakerConfig, SpliceConfig

config = PrefixSpeakerConfig(
    sample_rate=24_000,
    output_lead_ms=20.0,
    splice=SpliceConfig(
        crossfade_ms=30.0,
    ),
)
```
