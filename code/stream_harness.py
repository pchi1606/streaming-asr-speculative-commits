# Author: Prabhu Rangaraju
"""Streaming ASR harness: feeds a WAV through Vosk in fixed time chunks,
logging every partial hypothesis and the final result with timestamps."""
import json, wave, sys
from vosk import Model, KaldiRecognizer

def stream_file(wav_path, model, chunk_ms=200, sample_rate=16000):
    wf = wave.open(wav_path, "rb")
    assert wf.getnchannels() == 1 and wf.getsampwidth() == 2 and wf.getframerate() == sample_rate, \
        f"bad format: {wf.getparams()}"
    rec = KaldiRecognizer(model, sample_rate)
    rec.SetWords(True)
    chunk = int(sample_rate * chunk_ms / 1000)
    events = []  # (t_ms, kind, text)
    t = 0
    while True:
        data = wf.readframes(chunk)
        if not data:
            break
        t += chunk_ms
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            events.append((t, "segment", res.get("text", "")))
        else:
            pres = json.loads(rec.PartialResult())
            events.append((t, "partial", pres.get("partial", "")))
    fres = json.loads(rec.FinalResult())
    events.append((t, "final", fres.get("text", "")))
    wf.close()
    return events

if __name__ == "__main__":
    model = Model(sys.argv[1])
    for wav in sys.argv[2:]:
        print(f"=== {wav}")
        for t, kind, text in stream_file(wav, model):
            print(f"  t={t:5d}ms [{kind:7s}] {text!r}")
