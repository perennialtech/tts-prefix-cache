# tts-prefix-cache

Low-latency audio prefix splicing for speech and text-to-speech systems.

The core library is provider-agnostic. It works with generic speech audio:

1. Prepare and cache audio for the beginning of an utterance.
2. Generate or receive full utterance audio later.
3. Align the cached prefix against the full utterance using speech features and DTW.
4. Pick a splice boundary using alignment, energy, quiet-run, and sample-amplitude scoring.
5. Return a continuation, or create a stitched output with crossfade.

The important invariant is audio-level, not provider-level:

> The cached prefix audio must be an acoustic rendering of the beginning of the full utterance audio.

Same speaker, same engine, same settings, and same spoken prefix content produce the best results.

## Core usage

```py
from tts_prefix_cache import (
    SpliceConfig,
    prepare_prefix_audio,
    splice_from_full_audio,
)

config = SpliceConfig()
prepared = prepare_prefix_audio(prefix_audio, sample_rate=24000, config=config)

splice = splice_from_full_audio(
    prefix=prepared,
    full_audio=full_utterance_audio,
    sample_rate=24000,
    config=config,
)

continuation = splice.continuation
boundary = splice.boundary
```

## Offline stitching

```py
from tts_prefix_cache import stitch_audio

stitched_audio, splice = stitch_audio(
    prefix=prepared,
    full_audio=full_utterance_audio,
    sample_rate=24000,
    config=config,
)
```

## Streaming TTS helper

The optional `PrefixSpeaker` wrapper keeps TTS orchestration thin. It still does not care about providers.

```py
from tts_prefix_cache import BufferedWavSink, PrefixSpeaker, PrefixSpeakerConfig

speaker = PrefixSpeaker(
    tts=my_synthesizer,
    config=PrefixSpeakerConfig(sample_rate=24000),
)

sink = BufferedWavSink(path="out.wav", sample_rate=24000)

result = await speaker.speak(
    prefix="Sure, I can help with that.",
    rest=" First, open the settings menu.",
    sink=sink,
)

sink.save()
```

A synthesizer only needs to implement:

```py
async def synthesize(self, text: str, *, sample_rate: int) -> Audio:
    ...
```

Provider SDKs, auth, retries, formats, and decoding belong outside this package.

## FastAPI streaming

`PrefixSpeaker` also exposes async iterators for HTTP streaming. The package does not depend on FastAPI.

```py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from tts_prefix_cache import PrefixSpeaker, PrefixSpeakerConfig

app = FastAPI()


class SpeakRequest(BaseModel):
    prefix: str
    rest: str
    key: str | None = None


@app.on_event("startup")
async def startup() -> None:
    app.state.speaker = PrefixSpeaker(
        tts=my_synthesizer,
        config=PrefixSpeakerConfig(
            sample_rate=24000,
            pace_audio=False,
        ),
    )


@app.post("/speak.pcm")
async def speak_pcm(req: SpeakRequest):
    speaker: PrefixSpeaker = app.state.speaker

    return StreamingResponse(
        speaker.pcm16le_stream(
            prefix=req.prefix,
            rest=req.rest,
            key=req.key,
        ),
        media_type="audio/x-raw; format=S16LE; rate=24000; channels=1",
    )
```

For realtime sinks, keep `pace_audio=True`. For HTTP responses where the client/player handles buffering, `pace_audio=False` avoids artificial server-side playback pacing.
