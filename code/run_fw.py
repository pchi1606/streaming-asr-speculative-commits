# Author: Prabhu Rangaraju
"""Second engine: faster-whisper tiny.en (int8, CPU) in growing-buffer mode.
Every 750 ms the buffer [0:t] is re-decoded (greedy); the current hypothesis
is emitted as the unstable partial. This mirrors commercial streaming-ASR
semantics (revisable partials, LocalAgreement-style stabilization) without
any cloud service.
Stratified subset: 20 utterances per category (120 total), seeded.
Output: fw_events.jsonl with the same schema as run_vosk.py.
"""
import json, wave, random, os
import numpy as np
from faster_whisper import WhisperModel

STEP_MS = 750
SR = 16000
# Canonical frozen audio set; see run_vosk.py docstring and README.
AUDIO_DIR = os.environ.get("AUDIO_DIR", "audio/wav16k")

random.seed(20261004)
corpus = [json.loads(l) for l in open("corpus.jsonl")]
by_cat = {}
for r in corpus:
    by_cat.setdefault(r["category"], []).append(r)
subset = []
for c in sorted(by_cat):
    subset.extend(random.sample(by_cat[c], 20))
random.shuffle(subset)
print(f"subset: {len(subset)} utterances")

def load_wav(path):
    wf = wave.open(path, "rb")
    n = wf.getnframes()
    raw = wf.readframes(n)
    wf.close()
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

model = WhisperModel("models/fw-tiny-en", device="cpu", compute_type="int8")
done = set()
try:
    with open("fw_events.jsonl") as f:
        for line in f:
            done.add(json.loads(line)["id"])
except FileNotFoundError:
    pass
out = open("fw_events.jsonl", "a")
n_ev = 0
for i, row in enumerate(subset):
    if row["id"] in done:
        continue
    audio = load_wav(f"{AUDIO_DIR}/{row['id']}.wav")
    total_ms = int(len(audio) / SR * 1000)
    t = 0
    while t < total_ms:
        t += STEP_MS
        buf = audio[:int(t / 1000 * SR)]
        segs, _ = model.transcribe(buf, beam_size=1, vad_filter=False,
                                   word_timestamps=False)
        text = " ".join(s.text.strip() for s in segs).strip()
        out.write(json.dumps({"id": row["id"], "t_ms": min(t, total_ms),
                              "kind": "partial", "text": text, "words": []}) + "\n")
        n_ev += 1
    # final = decode of full buffer
    segs, _ = model.transcribe(audio, beam_size=1, vad_filter=False,
                               word_timestamps=False)
    text = " ".join(s.text.strip() for s in segs).strip()
    out.write(json.dumps({"id": row["id"], "t_ms": total_ms, "kind": "final",
                          "text": text, "words": []}) + "\n")
    n_ev += 1
    if (i + 1) % 20 == 0:
        print(f"fw streamed {i+1}/{len(subset)}", flush=True)
out.close()
json.dump([r["id"] for r in subset], open("fw_subset.json", "w"))
print(f"done: {len(subset)} utterances, {n_ev} events")
