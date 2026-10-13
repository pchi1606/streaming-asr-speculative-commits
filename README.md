# Reproduction package — speculative commits on streaming ASR partials

One-command reproduction of the results in *"Speculative Commits on
Streaming ASR Partials: Measuring Wasted-Action Rates for Voice Agents"*
(manuscript in preparation for SIGDIAL 2027).

**Author:** Prabhu Rangaraju

## One command

```bash
bash reproduce.sh
```

## What it does

1. Installs Python dependencies (`requirements.txt`; numpy, matplotlib,
   vosk, faster-whisper — CPU only).
2. Fetches the two ASR models if missing (`models/`; ~190 MB total).
3. Runs both streaming engines from the **frozen canonical audio**
   (`data/audio/`, 360 WAVs): Vosk partial-hypothesis streaming and the
   faster-whisper growing-buffer re-decoder.
4. Runs the analysis (`code/analyze.py`): policy simulation, paired
   bootstrap 95% CIs (B=2000, seeded), cause decomposition,
   field-error profile.
5. Regenerates the four manuscript figures (`figures/`).
6. Verifies headline numbers (`scripts/verify_headlines.py`) and prints
   PASS/FAIL per check.

Expected runtime: ~3–4 hours on a 2-CPU box (ASR decoding dominates);
steps 4–6 alone take under 5 minutes. Outputs land in the package root
(`vosk_events.jsonl`, `fw_events.jsonl`, `results_*.json`) and
`figures/`.

## What is verified

- faster-whisper results are sha256-identical to `expected/` (fully
  deterministic given frozen audio).
- Vosk results are sha256-identical, or — on a rare Vosk final-decode
  flip — every headline lies within the canonical 95% CIs and vice versa
  (mutual containment).
- Manuscript headlines: Vosk P2 K=2 WAR 38.6% [33.4, 44.1], 897.9 ms
  saved; threshold-paradox operating curves; incompleteness-vs-instability
  cause counts (98 vs 14); per-category WARs; P3 action rates; FW
  speculation-collapse numbers.

## Determinism statement

Two known, documented sources of run-to-run variation (both disclosed in
the manuscript's Limitations):

1. **TTS synthesis is stochastic** (Piper exposes no seed), so audio is
   never re-synthesized here: reproduction starts at the frozen canonical
   WAVs (`data/audio/`), which produced every reported number.
2. **Vosk's `FinalResult()` lattice decode flips rarely across process
   runs** (observed 1/360 utterances across two full runs). Partial
   hypotheses are bit-identical; all headlines stay within the reported
   95% CIs. The verifier's mutual-CI fallback encodes exactly this
   criterion.

*2026-10-12 note: a second full engine re-run on the frozen audio
(Vosk 360/360, faster-whisper 120/120) confirmed this. Faster-whisper
results were sha256-identical; Vosk had exactly one final-decode flip
(inclusion 290→291) with every headline point estimate inside the
reported 95% CIs. No manuscript numbers changed.*

## Data

- `data/corpus.jsonl`: 360 scripted utterances with ground-truth tool
  calls (categories A 60 / B 90 / C 60 / D 60 / E 45 / F 45). Synthetic,
  seeded generator — no human subjects, no third-party data.
- `data/audio/`: the 360 canonical 16 kHz mono WAVs (frozen 2026-10-05).
- `data/fw_subset.json`: the stratified 120-utterance subset (20 per
  category) used for the faster-whisper engine.
- `expected/`: canonical `results_vosk.json` / `results_fw.json` with
  SHA256SUMS.

## Models

- `vosk-model-small-en-us-0.15` (Alpha Cephei; model Apache-2.0),
  fetched from https://alphacephei.com/vosk/models/
- faster-whisper `tiny.en` (SYSTRAN; MIT), fetched via the
  faster-whisper downloader
- Piper `en_US-lessac-medium` voice was used only for the original
  synthesis step, which is excluded from reproduction by design

## Citation

See `CITATION.md`.
