# Citation

## Paper

Prabhu Rangaraju. *Speculative Commits on Streaming ASR Partials:
Measuring Wasted-Action Rates for Voice Agents.* Manuscript in
preparation for SIGDIAL 2027 (28th Annual Meeting of the Special
Interest Group on Discourse and Dialogue). 2026.

## Code

The experiment code in this package (`code/`, `scripts/`,
`reproduce.sh`) was authored by Prabhu Rangaraju for this paper.

## Data and models

- Corpus (`data/corpus.jsonl`) and audio (`data/audio/`): synthetic,
  generated for this paper; no human subjects, no third-party data.
- Vosk `vosk-model-small-en-us-0.15` — Alpha Cephei Inc.
  https://alphacephei.com/vosk/ (model: Apache-2.0)
- faster-whisper `tiny.en` — SYSTRAN
  https://github.com/SYSTRAN/faster-whisper (MIT)
- Piper `en_US-lessac-medium` text-to-speech voice — used only for the
  original (non-reproduced) synthesis step
