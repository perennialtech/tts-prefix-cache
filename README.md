# tts-prefix-cache

Low-latency audio prefix splicing for speech and TTS pipelines.

`tts-prefix-cache` lets you play a cached synthesized prefix immediately while a full utterance is being generated in the background, then splice into the generated continuation once it is ready.

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
        rest="I can help with that.",
    )

    write_wav("out/response.wav", audio, sample_rate=24_000)
    print(result)


asyncio.run(main())
```

## Live playback

Use `speak()` when you are feeding a live sink. Live playback has to make timing decisions: when to release the held prefix tail, when to splice, and when to inject same-stream silence if full synthesis is late.

```python
from tts_prefix_cache import BufferedWavSink, PrefixSpeakerConfig

config = PrefixSpeakerConfig(
    sample_rate=24_000,
    playback_clock="source",
)

speaker = PrefixSpeaker(tts=MySynthesizer(), config=config)
sink = MyLiveAudioSink()

result = await speaker.speak(
    prefix="Sure, ",
    rest="I can help with that.",
    sink=sink,
)
```

## Playback clock modes

The speaker needs exactly one playback clock. The clock controls when cached audio is considered "heard enough" for splice and silence decisions.

### `source`

```python
PrefixSpeakerConfig(playback_clock="source")
```

This is the default and safest mode.

The library writes cached prefix audio in real time using source-side sleeps. This keeps splice and silence decisions aligned with ordinary wall-clock playback.

Use this for:

- `audio_stream()`
- simple queues
- WebSocket senders without real playout feedback
- in-memory sinks
- sinks that only copy or enqueue data

### `buffered_timeline`

```python
PrefixSpeakerConfig(
    playback_clock="buffered_timeline",
    output_lead_ms=20.0,
)
```

This mode writes already-known audio as fast as the sink accepts it, but still tracks a local playout timeline.

The speaker does not inject silence merely because Python finished writing cached prefix bytes quickly. It waits until the queued known audio is close to running out, then decides whether to splice, release the held tail, or add silence.

Use this when the downstream can buffer and pace playback but does not provide direct playback-clock backpressure.

### `sink`

```python
PrefixSpeakerConfig(playback_clock="sink")
```

This mode assumes `sink.write()` is the clock.

The speaker does not sleep for cached audio or silence. `sink.write()` must block or apply backpressure according to real playback capacity. Do not use this mode with ordinary in-memory buffers, normal WebSocket sends, or unbounded queues.

There should be one clock, not source sleeps plus sink pacing.

## Streaming audio chunks

Use `audio_stream()` when you want NumPy audio chunks:

```python
async for chunk in speaker.audio_stream(
    prefix="Sure, ",
    rest="I can help with that.",
):
    send_audio_chunk(chunk)
```

Use `pcm16le_stream()` when your transport expects raw little-endian 16-bit PCM bytes:

```python
async for data in speaker.pcm16le_stream(
    prefix="Sure, ",
    rest="I can help with that.",
):
    send_bytes(data)
```

`audio_stream()` uses the same playback clock configured on the speaker.

## Stable cache identifiers

If your application has its own stable cache identifiers, pass a key:

```python
await speaker.speak(
    prefix="Sure, ",
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

Then pass it to `speak()`:

```python
await speaker.speak(
    prefix="Sure, ",
    rest="I can help with that.",
    sink=MySink(),
)
```

The sink contract depends on `playback_clock`:

- With `"source"`, `write()` may simply copy, enqueue, or send data.
- With `"buffered_timeline"`, `write()` may accept data faster than playback, but the downstream should handle playback pacing.
- With `"sink"`, `write()` must provide playback-clock backpressure.

## Configuration

Runtime behavior is configured with `PrefixSpeakerConfig` and its nested splice configuration.

Prefer constructing those dataclasses directly in code so your editor and type checker can use the package's type definitions as the source of truth:

```python
from tts_prefix_cache import PrefixSpeakerConfig, SpliceConfig

config = PrefixSpeakerConfig(
    sample_rate=24_000,
    playback_clock="buffered_timeline",
    output_lead_ms=20.0,
    splice=SpliceConfig(
        crossfade_ms=30.0,
    ),
)
```
