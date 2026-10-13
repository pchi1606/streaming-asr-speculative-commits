# Author: Prabhu Rangaraju
"""Full Vosk streaming run over the 360-utterance corpus.
Feeds each WAV in 200 ms chunks; logs every partial (with word confidences
when available) and the final (with word confidences + timings).
Output: vosk_events.jsonl — one JSON object per event:
  {id, t_ms, kind: partial|segment|final, text, words:[{word,conf,start,end}]}
Deterministic except for rare Vosk final-decode flips: single-threaded,
fixed chunk size, seeded model. Given byte-identical input WAVs, partial
hypotheses are bit-identical across runs; the final transcript flipped on
1/360 utterances across two full runs (all headlines stay within the
reported 95% CIs — see README "Determinism statement").
Reproduction starts from the frozen canonical audio set, never from
re-synthesis (Piper TTS is run-to-run stochastic; see README).
Set AUDIO_DIR to point at the canonical audio directory
(default: audio/wav16k).
"""
import json, wave, sys, os
from vosk import Model, KaldiRecognizer

CHUNK_MS = 200
SR = 16000
AUDIO_DIR = os.environ.get("AUDIO_DIR", "audio/wav16k")

def stream_file(wav_path, model):
    wf = wave.open(wav_path, "rb")
    assert wf.getnchannels() == 1 and wf.getsampwidth() == 2 \
        and wf.getframerate() == SR, wf.getparams()
    rec = KaldiRecognizer(model, SR)
    rec.SetWords(True)
    chunk = int(SR * CHUNK_MS / 1000)
    events = []
    t = 0
    while True:
        data = wf.readframes(chunk)
        if not data:
            break
        t += CHUNK_MS
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            events.append({"t_ms": t, "kind": "segment", "text": res.get("text", ""),
                           "words": res.get("result", [])})
        else:
            pres = json.loads(rec.PartialResult())
            events.append({"t_ms": t, "kind": "partial", "text": pres.get("partial", ""),
                           "words": pres.get("result", [])})
    fres = json.loads(rec.FinalResult())
    events.append({"t_ms": t, "kind": "final", "text": fres.get("text", ""),
                   "words": fres.get("result", [])})
    wf.close()
    return events

def main():
    model = Model("models/vosk-model-small-en-us-0.15")
    done = set()
    try:
        with open("vosk_events.jsonl") as f:
            for line in f:
                done.add(json.loads(line)["id"])
    except FileNotFoundError:
        pass
    out = open("vosk_events.jsonl", "a")
    n_ev = 0
    with open("corpus.jsonl") as f:
        rows = [json.loads(l) for l in f]
    if "--reverse" in sys.argv:
        rows = rows[::-1]
    for i, row in enumerate(rows):
        # live done-check: refresh from disk each iteration so parallel
        # workers never duplicate an utterance
        with open("vosk_events.jsonl") as _f:
            _done = set(json.loads(_l)["id"] for _l in _f)
        if row["id"] in _done:
            continue
        wav = f"{AUDIO_DIR}/{row['id']}.wav"
        for ev in stream_file(wav, model):
            ev["id"] = row["id"]
            out.write(json.dumps(ev) + "\n")
            n_ev += 1
        if (i + 1) % 60 == 0:
            print(f"streamed {i+1}/{len(rows)}", flush=True)
    out.close()
    print(f"done: {len(rows)} utterances, {n_ev} events")

if __name__ == "__main__":
    main()
