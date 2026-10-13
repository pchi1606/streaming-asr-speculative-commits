# Author: Prabhu Rangaraju
"""Action parser + agent-policy simulator for S2.

Parser: deterministic fuzzy slot extractor over transcript text.
  parse(text) -> {"intent": str, "slots": {...}} | None (incomplete/unparseable)
  Semantics mirror the template generator: last non-negated mention wins;
  "not X" excludes X; "copy Y" marks Y as cc rather than recipient.

Policies (applied to a timeline of (t_ms, text) hypotheses):
  P1 act-on-final: commit parse(final) at t_final. Waste-free by construction.
  P2(K) stability-thresholded speculative commit: commit at the first t where
      parse(text_t) is complete and identical to the parsed actions of the
      previous K-1 consecutive hypotheses (temporal agreement = engine-agnostic
      confidence proxy; Vosk partials do not expose per-word confidences).
  P3(K) commit-with-rollback: P2 commit rule, then every later hypothesis that
      parses to a different complete action triggers a rollback + re-commit.

Metrics per utterance: wasted action (committed != ground truth), cause
(incompleteness vs ASR-instability via word-prefix test), latency saved,
action counts, final correctness.
"""
import json, re, difflib
from collections import Counter

DESTS = ["boston", "chicago", "seattle", "denver", "austin", "miami"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
ACCOUNTS = ["savings", "checking"]
NAMES = ["alice", "bob", "carol", "dave", "erin", "frank"]
TIMES = ["seven a m", "six p m", "eight a m", "five p m", "nine a m", "four p m"]
AMOUNTS = ["two hundred", "five hundred", "one thousand", "fifty", "three hundred"]
ALL_AMOUNTS = AMOUNTS + ["fifteen", "thirteen", "thirty", "fourteen", "forty",
                         "sixteen", "sixty", "nineteen", "ninety"]
TODS = ["morning", "evening"]
HOURS = ["three", "four", "five"]

_ONES = ["zero","one","two","three","four","five","six","seven","eight",
          "nine","ten","eleven","twelve","thirteen","fourteen","fifteen",
          "sixteen","seventeen","eighteen","nineteen"]
_TENS = {20:"twenty",30:"thirty",40:"forty",50:"fifty",60:"sixty",
        70:"seventy",80:"eighty",90:"ninety"}
def _numwords(n):
    if n < 20:
        return [_ONES[n]]
    if n < 100:
        t, o = (n//10)*10, n % 10
        return [_TENS[t]] + ([_ONES[o]] if o else [])
    if n < 1000:
        h, r = n//100, n % 100
        return [_ONES[h], "hundred"] + (_numwords(r) if r else [])
    if n == 1000:
        return ["one", "thousand"]
    return [str(n)]  # fallback: leave as digits

def toks(text):
    # strip thousand-separators: "$1,000" -> "$1000"
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    raw = re.findall(r"\$?\d+(?:am|pm)?|[a-z']+", text)
    out = []
    i = 0
    while i < len(raw):
        w = raw[i]
        # Vosk emits "nine am"; templates use "nine a m" — normalize
        if w == "am":
            out.extend(["a", "m"])
        elif w == "pm":
            out.extend(["p", "m"])
        elif w in ("o'clock", "oclock"):
            pass  # dropped; the hour digit carries the meaning
        elif w.startswith("$") and w[1:].isdigit():
            out.extend(_numwords(int(w[1:])) + ["dollars"])
        elif re.fullmatch(r"\d+(am|pm)", w):
            m = re.fullmatch(r"(\d+)(am|pm)", w)
            out.extend(_numwords(int(m.group(1))))
            out.extend(["a", "m"] if m.group(2) == "am" else ["p", "m"])
        elif w.isdigit():
            out.extend(_numwords(int(w)))
        else:
            out.append(w)
        i += 1
    return out

def tok_ratio(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

def find_all(t, vocab, cutoff=0.82):
    """All (pos, value) matches of vocab items in token list t (fuzzy)."""
    hits = []
    for i, w in enumerate(t):
        best, bv = 0.0, None
        for v in vocab:
            vt = v.split()
            if len(vt) == 1:
                r = tok_ratio(w, v)
                if r > best:
                    best, bv = r, v
            else:
                seg = t[i:i+len(vt)]
                if len(seg) < len(vt):
                    continue
                r = sum(tok_ratio(s, x) for s, x in zip(seg, vt)) / len(vt)
                # require each token decent
                if all(tok_ratio(s, x) >= 0.72 for s, x in zip(seg, vt)) and r > best:
                    best, bv = r, v
        if best >= cutoff:
            hits.append((i, bv))
    return hits

def is_negated(t, pos):
    """True if the match at token pos is immediately preceded by 'not',
    allowing one intervening word (e.g. 'not my checking')."""
    for j in (pos - 1, pos - 2):
        if j >= 0 and tok_ratio(t[j], "not") >= 0.85:
            return True
    return False

def intent_of(t):
    scores = Counter()
    def has(vocab, cutoff=0.82):
        return bool(find_all(t, vocab, cutoff))
    if has(["flight"]): scores["book_flight"] += 2
    if has(["book"]): scores["book_flight"] += 1
    if has(["transfer"]): scores["transfer"] += 2
    if has(["dollar", "dollars"]): scores["transfer"] += 1
    if has(["account"]): scores["transfer"] += 1
    if has(["send"]): scores["send_report"] += 2
    if has(["report"]): scores["send_report"] += 1
    if has(["alarm"]): scores["set_alarm"] += 2
    if has(["remind", "reminder"]): scores["reminder"] += 2
    if has(["schedule"]): scores["schedule_meeting"] += 2
    if has(["meeting"]): scores["schedule_meeting"] += 1
    # transfer_alarm needs BOTH transfer and alarm cues
    if has(["alarm"]) and has(["transfer"]):
        scores["transfer_alarm"] += 5
    if not scores:
        return None
    return scores.most_common(1)[0][0]

def last_valid(t, hits):
    cands = [(p, v) for p, v in hits if not is_negated(t, p)]
    return cands[-1][1] if cands else None

# Mandatory slots per intent: a hypothesis only parses to a "complete"
# action when all of these are present. Optional refinements (tod for
# book_flight, except for reminder, cc for send_report, day for transfer)
# may arrive later in the utterance — that late arrival is exactly the
# speculative-commit hazard under study, so their absence must NOT block
# a commit.
REQ = {
    "book_flight": {"dest", "day"},
    "transfer": {"amount", "account"},
    "send_report": {"recipient"},
    "set_alarm": {"time"},
    "reminder": {"task", "time"},
    "schedule_meeting": {"attendees", "time", "day"},
    "transfer_alarm": {"amount", "time"},
}

def parse(text):
    t = toks(text)
    if not t:
        return None
    intent = intent_of(t)
    if intent is None:
        return None
    slots = {}
    if intent == "book_flight":
        d = last_valid(t, find_all(t, DESTS))
        day = last_valid(t, find_all(t, DAYS))
        if not d or not day:
            return None
        slots = {"dest": d, "day": day}
        tod = last_valid(t, find_all(t, TODS))
        if tod: slots["tod"] = tod
        names = [v for _, v in find_all(t, NAMES)]
        if len(names) >= 2: slots["passengers"] = sorted(set(names))
    elif intent == "transfer":
        a = last_valid(t, find_all(t, ALL_AMOUNTS))
        acc = last_valid(t, find_all(t, ACCOUNTS))
        if not a or not acc:
            return None
        slots = {"amount": a, "account": acc}
        day = last_valid(t, find_all(t, DAYS))
        if day: slots["day"] = day
    elif intent == "send_report":
        names = [(p, v) for p, v in find_all(t, NAMES) if not is_negated(t, p)]
        if not names:
            return None
        has_copy = bool(find_all(t, ["copy"]))
        if has_copy and len(names) >= 2:
            slots = {"recipient": names[0][1], "cc": names[-1][1]}
        else:
            slots = {"recipient": names[-1][1]}
    elif intent == "set_alarm":
        tm = last_valid(t, find_all(t, TIMES))
        if not tm:
            return None
        slots = {"time": tm}
    elif intent == "reminder":
        if find_all(t, ["water"]):
            slots = {"task": "water the plants"}
        else:
            names = [(p, v) for p, v in find_all(t, NAMES) if not is_negated(t, p)]
            if not names:
                return None
            slots = {"task": f"call {names[-1][1]}"}
        tm = last_valid(t, find_all(t, TIMES))
        if tm: slots["time"] = tm
        exc = last_valid(t, find_all(t, DAYS))
        # 'except' day: token before it is 'except'
        if exc:
            slots["except"] = exc
    elif intent == "schedule_meeting":
        names = sorted(set(v for _, v in find_all(t, NAMES)))
        tm = last_valid(t, find_all(t, TIMES))
        day = last_valid(t, find_all(t, DAYS))
        if not names or not tm or not day:
            return None
        slots = {"attendees": names, "time": tm, "day": day}
    elif intent == "transfer_alarm":
        a = last_valid(t, find_all(t, ALL_AMOUNTS))
        h = last_valid(t, find_all(t, HOURS))
        if not a or not h:
            return None
        slots = {"amount": a, "time": h}
    if intent == "reminder":
        # two template variants: {task,time} and {task,except}
        if not ({"task", "time"}.issubset(slots.keys()) or
                {"task", "except"}.issubset(slots.keys())):
            return None
    elif not REQ[intent].issubset(slots.keys()):
        return None
    return {"intent": intent, "slots": slots}

def action_eq(a, b):
    if a is None or b is None:
        return False
    if a["intent"] != b["intent"]:
        return False
    sa, sb = a["slots"], b["slots"]
    if set(sa) != set(sb):
        return False
    for k in sa:
        va, vb = sa[k], sb[k]
        if isinstance(va, list):
            if sorted(va) != sorted(vb):
                return False
        elif va != vb:
            return False
    return True

def is_word_prefix(short, long):
    st, lt = toks(short), toks(long)
    return len(st) <= len(lt) and lt[:len(st)] == st

def simulate(timeline, gt_action, K):
    """timeline: list of (t_ms, text) hypotheses ending with final.
    Returns dict of policy outcomes."""
    final_t, final_text = timeline[-1]
    parses = [(t, parse(x)) for t, x in timeline]
    out = {"t_final": final_t, "final_text": final_text,
           "final_parse": parses[-1][1]}
    # P1
    out["p1"] = {"commit_t": final_t, "action": parses[-1][1],
                 "wasted": 0, "saved_ms": 0}
    # P2: first t with complete parse stable over K consecutive hypotheses
    commit = None
    for i in range(len(parses)):
        t, a = parses[i]
        if a is None:
            continue
        if i + 1 >= K and all(parses[j][1] is not None and
                              action_eq(parses[j][1], a)
                              for j in range(i - K + 1, i + 1)):
            commit = (t, a, timeline[i][1])
            break
    if commit is None:
        t, a, txt = final_t, parses[-1][1], final_text
    else:
        t, a, txt = commit
    wasted = 0 if action_eq(a, gt_action) else 1
    cause = None
    if wasted and commit is not None:
        cause = ("incompleteness" if is_word_prefix(txt, final_text)
                 else "instability")
    out["p2"] = {"commit_t": t, "action": a, "wasted": wasted,
                 "cause": cause, "saved_ms": final_t - t,
                 "speculative": commit is not None}
    # P3: P2 commit, then rollbacks on change
    actions = []
    if commit is not None:
        actions.append(("commit", commit[0], commit[1]))
        cur = commit[1]
        for j in range(i + 1, len(parses)):
            tj, aj = parses[j]
            if aj is not None and not action_eq(aj, cur):
                actions.append(("rollback+commit", tj, aj))
                cur = aj
    else:
        actions.append(("commit", final_t, parses[-1][1]))
    first = actions[0]
    out["p3"] = {"actions": [(k, tt) for k, tt, _ in actions],
                 "n_actions": len(actions),
                 "first_wasted": 0 if action_eq(first[2], gt_action) else 1,
                 "final_correct": 1 if action_eq(actions[-1][2], gt_action) else 0,
                 "saved_ms": final_t - first[1]}
    return out
