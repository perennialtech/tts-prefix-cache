# tts-prefix-cache

Low-latency audio prefix splicing for speech and TTS pipelines.

`tts-prefix-cache` lets you play a cached synthesized prefix immediately while a full utterance is generated in the background, then splice into the generated continuation once it is ready.

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

Implement the `Synthesizer` protocol exported by the package. Its `synthesize` method is called with text and a requested sample rate.

Return mono, one-dimensional, finite `numpy.float32` audio at that sample rate. Samples are expected to be normal floating-point PCM-style values. Use `pcm16le_bytes()` when you need PCM16LE output bytes.

## Live playback

Use `speak()` when you are feeding a live sink.

```python
from tts_prefix_cache import PrefixSpeaker, PrefixSpeakerConfig

config = PrefixSpeakerConfig(
    sample_rate=24_000,
)

speaker = PrefixSpeaker(tts=MySynthesizer(), config=config)
sink = MyLiveAudioSink()

result = await speaker.speak(
    prefix="Sure, ",
    continuation="I can help with that.",
    sink=sink,
)
```

A sink only needs an async `write` method that accepts an audio chunk:

```python
class MySink:
    async def write(self, chunk) -> None:
        ...
```

The sink should enqueue, copy, send, or otherwise hand off audio. It should not sleep to simulate playback. Downstream audio infrastructure should handle actual playback pacing.

## Offline rendering

Use `render()` when you want a complete audio array, for example to write a WAV file or run offline tests.

```python
audio, result = await speaker.render(
    prefix="Sure, ",
    continuation="I can help with that.",
)
```

Offline rendering does not simulate live playback.

## Structured events and results

Pass a logger callback to receive structured `PrefixSpeakerEvent` objects for diagnostics, metrics, and debugging.

```python
from tts_prefix_cache import PrefixSpeaker, PrefixSpeakerEvent


def log_event(event: PrefixSpeakerEvent) -> None:
    print(event.name, event.data)


speaker = PrefixSpeaker(
    tts=MySynthesizer(),
    logger=log_event,
)
```

`speak()` and `render()` return result objects with operation metadata. Treat the exported dataclass definitions as the source of truth for available fields.

## Cache keys and correctness

For ordinary per-speaker, process-local caching, you can omit `cache_key`.

Pass `cache_key` explicitly when a cache is shared across speakers, restarts, workers, or processes, or whenever the same cache may be used with different synthesis inputs.

```python
prefix = "Sure, "

await speaker.speak(
    prefix=prefix,
    continuation="I can help with that.",
    cache_key=(
        "provider-a",
        "model-x",
        "voice-a",
        speaker.config.sample_rate,
        "default-style",
        speaker.config.splice,
        prefix,
    ),
    sink=sink,
)
```

The cache key must identify every input that can change either the prefix waveform or the prepared prefix data, including the exact prefix text and all relevant synthesis and splice settings.

Do not reuse a cache key across different voices, models, sample rates, style settings, or splice configurations. A wrong cache key can splice prefix audio from one synthesis configuration into full audio from another, producing bad alignment, audible jumps, or incorrect playback.

## Prefix choice

The cached prefix is synthesized by itself, while the full utterance is synthesized as `prefix + continuation`.

Some TTS systems render a standalone phrase differently from the same phrase at the start of a longer sentence. The library aligns and crossfades the audio, but best results come from prefixes whose standalone rendering is close to their rendering inside the full utterance.

Stable short prefixes such as "Sure, " or "One moment, " usually work better than prefixes whose standalone rendering strongly depends on the following words.

## Configuration

Runtime behavior is configured with `PrefixSpeakerConfig` and `SpliceConfig`.

Prefer constructing those dataclasses directly in code so your editor and type checker can use the package's type definitions as the source of truth.

```python
from tts_prefix_cache import PrefixSpeakerConfig, SpliceConfig

config = PrefixSpeakerConfig(
    sample_rate=24_000,
    splice=SpliceConfig(
        crossfade_ms=30.0,
    ),
)
```
