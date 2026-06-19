# tts-prefix-cache

Low-latency helper for text-to-speech systems where a stable prefix can be cached and played immediately while the full utterance is synthesized in the background.

The library:

1. Synthesizes and caches a prefix clip.
2. Starts full utterance synthesis in parallel.
3. Streams the cached prefix to an audio sink.
4. Inserts same-stream silence if the full synthesis is not ready yet.
5. Finds a splice boundary in the full audio using DTW plus nearby silence detection.
6. Streams the generated continuation.
