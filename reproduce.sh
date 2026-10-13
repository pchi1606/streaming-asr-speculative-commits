#!/usr/bin/env bash
# Author: Prabhu Rangaraju
# One-command reproduction of the S2 streaming-ASR speculative-commit results.
#
# Usage:  bash reproduce.sh
# Run from this directory. Requires Python 3.10+ and pip.
#
# What it does:
#   1. installs Python dependencies (requirements.txt)
#   2. fetches the two ASR models if missing (Vosk small-en-us-0.15,
#      faster-whisper tiny.en) into models/
#   3. runs both streaming engines from the FROZEN canonical audio
#      (data/audio/ — 360 WAVs; synthesis is deliberately NOT re-run,
#      see README "Determinism statement")
#        code/run_vosk.py  -> vosk_events.jsonl   (~2.5 h on 2 CPUs)
#        code/run_fw.py    -> fw_events.jsonl     (~1 h on 2 CPUs)
#   4. runs the analysis (code/analyze.py) -> results_vosk.json,
#      results_fw.json (seeded bootstrap, deterministic)
#   5. regenerates the four manuscript figures (code/make_figures.py)
#   6. verifies headline numbers against expected/ and reports PASS/FAIL
#
# Expected runtime: ~3-4 h on a 2-CPU box (ASR decoding dominates);
# steps 4-6 alone take under 5 minutes.
set -euo pipefail
cd "$(dirname "$0")"

echo "================================================================"
echo " STEP 1/6: dependencies"
echo "================================================================"
pip install -r requirements.txt || echo "WARNING: pip install failed; continuing with installed dependencies."

echo "================================================================"
echo " STEP 2/6: models (download if missing)"
echo "================================================================"
mkdir -p models
if [ ! -d models/vosk-model-small-en-us-0.15 ]; then
  echo "Fetching vosk-model-small-en-us-0.15 (~40 MB)..."
  curl -L -o /tmp/vosk-small.zip \
    https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
  unzip -q -o /tmp/vosk-small.zip -d models/
  rm -f /tmp/vosk-small.zip
else
  echo "Vosk model present; skipping download."
fi
if [ ! -d models/fw-tiny-en ]; then
  echo "Fetching faster-whisper tiny.en (~150 MB)..."
  python3 -c "
from faster_whisper import WhisperModel
WhisperModel('tiny.en', device='cpu', compute_type='int8',
             download_root='models/_dl')
print('tiny.en downloaded')
"
  SNAP=$(find models/_dl -name model.bin | head -1 | xargs dirname)
  mkdir -p models/fw-tiny-en
  cp "$SNAP"/model.bin "$SNAP"/config.json \
     "$SNAP"/tokenizer.json "$SNAP"/vocabulary.txt models/fw-tiny-en/
  rm -rf models/_dl
else
  echo "faster-whisper tiny.en present; skipping download."
fi

echo "================================================================"
echo " STEP 3/6: streaming engines (from frozen canonical audio)"
echo "================================================================"
export AUDIO_DIR="$PWD/data/audio"
# The pipeline scripts read corpus.jsonl / fw_subset.json from the
# working directory; stage them from data/ (deterministic inputs).
cp data/corpus.jsonl data/fw_subset.json .
python3 code/run_vosk.py
python3 code/run_fw.py

echo "================================================================"
echo " STEP 4/6: analysis"
echo "================================================================"
python3 code/analyze.py vosk
python3 code/analyze.py fw

echo "================================================================"
echo " STEP 5/6: figures"
echo "================================================================"
mkdir -p figures
python3 code/make_figures.py

echo "================================================================"
echo " STEP 6/6: verify headline numbers"
echo "================================================================"
python3 scripts/verify_headlines.py
