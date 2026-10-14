"""Voice profiles: enroll known reps once, then recognise them in any call.

Enroll a rep from a stretch of their audio:
    python profiles.py enroll "Jane Doe" data/audio/sample-1.mp3 --start 34 --end 120
Enroll from a diarized speaker (uses all of that speaker's turns):
    python profiles.py enroll-speaker "Jane Doe" output/sample-1.speakers.json SPEAKER_01
Who is speaking in a call:
    python profiles.py identify output/sample-1.speakers.json
List / delete:
    python profiles.py list
    python profiles.py delete "Jane Doe"

Profiles are voice biometrics (personal data): they live in ./profiles, which is
git-ignored. Only enroll people who have consented.
"""
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sherpa_onnx

from diarize import ROOT, EMB_MODEL, load_audio

PROFILE_DIR = ROOT / "profiles"
SR = 16000
MAX_SECONDS = 60.0
MIN_SECONDS = 5.0
DEFAULT_THRESHOLD = 0.5  # cosine similarity; tune on real calls


def _extractor():
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMB_MODEL), num_threads=4)
    if not cfg.validate():
        sys.exit("invalid embedding model config")
    return sherpa_onnx.SpeakerEmbeddingExtractor(cfg)


def embed(samples: np.ndarray, extractor=None) -> np.ndarray:
    ex = extractor or _extractor()
    stream = ex.create_stream()
    stream.accept_waveform(sample_rate=SR, waveform=samples)
    stream.input_finished()
    v = np.array(ex.compute(stream), dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def cosine(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def audio_for_turns(samples: np.ndarray, turns, max_seconds=MAX_SECONDS) -> np.ndarray:
    """Concatenate (start, end) turns, longest first, up to max_seconds."""
    parts, total = [], 0.0
    for s, e in sorted(turns, key=lambda t: t[0] - t[1]):
        if total >= max_seconds:
            break
        chunk = samples[int(s * SR) : int(e * SR)]
        if len(chunk) < 0.8 * SR:
            continue
        parts.append(chunk)
        total += len(chunk) / SR
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def save_profile(name: str, vec: np.ndarray, seconds: float, role: str = "salesperson") -> Path:
    PROFILE_DIR.mkdir(exist_ok=True)
    path = PROFILE_DIR / f"{slug(name)}.json"
    path.write_text(
        json.dumps(
            {
                "name": name,
                "role": role,
                "seconds": round(seconds, 1),
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "embedding": [round(float(x), 6) for x in vec],
            }
        ),
        encoding="utf-8",
    )
    return path


def load_profiles():
    if not PROFILE_DIR.exists():
        return []
    out = []
    for f in sorted(PROFILE_DIR.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        d["embedding"] = np.array(d["embedding"], dtype=np.float32)
        out.append(d)
    return out


def identify_speakers(audio_path: str, speaker_turns, threshold=DEFAULT_THRESHOLD, profiles=None):
    """Return {speaker: {name, role, score, matched, seconds}} for each diarized speaker."""
    profiles = profiles if profiles is not None else load_profiles()
    samples = load_audio(audio_path, SR)
    ex = _extractor()
    by_spk = {}
    for t in speaker_turns:
        by_spk.setdefault(t["speaker"], []).append((t["start"], t["end"]))

    result = {}
    for spk, turns in sorted(by_spk.items()):
        clip = audio_for_turns(samples, turns)
        secs = len(clip) / SR
        if secs < MIN_SECONDS or not profiles:
            result[spk] = {"name": None, "role": None, "score": None, "matched": False, "seconds": round(secs, 1)}
            continue
        v = embed(clip, ex)
        scored = sorted(((cosine(v, p["embedding"]), p) for p in profiles), key=lambda x: -x[0])
        score, best = scored[0]
        matched = score >= threshold
        result[spk] = {
            "name": best["name"] if matched else None,
            "role": best["role"] if matched else None,
            "score": round(score, 3),
            "closest": best["name"],
            "matched": matched,
            "seconds": round(secs, 1),
        }
    return result


def cmd_enroll(a):
    samples = load_audio(a.audio, SR)
    clip = samples[int(a.start * SR) : int((a.end if a.end else len(samples) / SR) * SR)]
    if len(clip) / SR < MIN_SECONDS:
        sys.exit(f"need at least {MIN_SECONDS:.0f}s of clean speech from one person")
    path = save_profile(a.name, embed(clip), len(clip) / SR, a.role)
    print(f"enrolled {a.name!r} from {len(clip) / SR:.0f}s -> {path}")


def cmd_enroll_speaker(a):
    data = json.loads(Path(a.transcript).read_text(encoding="utf-8"))
    audio = a.audio or data["audio"]
    turns = [(t["start"], t["end"]) for t in data["speaker_turns"] if t["speaker"] == a.speaker]
    if not turns:
        sys.exit(f"no turns for {a.speaker}; have {data['speakers']}")
    clip = audio_for_turns(load_audio(audio, SR), turns)
    if len(clip) / SR < MIN_SECONDS:
        sys.exit("not enough speech from that speaker")
    path = save_profile(a.name, embed(clip), len(clip) / SR, a.role)
    print(f"enrolled {a.name!r} from {len(clip) / SR:.0f}s of {a.speaker} -> {path}")


def cmd_identify(a):
    data = json.loads(Path(a.transcript).read_text(encoding="utf-8"))
    res = identify_speakers(a.audio or data["audio"], data["speaker_turns"], a.threshold)
    if not load_profiles():
        print("no profiles enrolled yet")
    for spk, r in res.items():
        who = r["name"] or "unknown"
        print(f"{spk}: {who:15s} score={r['score']}  closest={r.get('closest')}  speech={r['seconds']}s")


def cmd_list(_):
    for p in load_profiles():
        print(f"{p['name']:20s} role={p['role']:12s} {p['seconds']}s  {p['created']}")


def cmd_delete(a):
    f = PROFILE_DIR / f"{slug(a.name)}.json"
    if f.exists():
        f.unlink()
        print(f"deleted {a.name!r}")
    else:
        print("no such profile")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enroll")
    e.add_argument("name")
    e.add_argument("audio")
    e.add_argument("--start", type=float, default=0.0)
    e.add_argument("--end", type=float)
    e.add_argument("--role", default="salesperson")
    e.set_defaults(fn=cmd_enroll)

    es = sub.add_parser("enroll-speaker")
    es.add_argument("name")
    es.add_argument("transcript")
    es.add_argument("speaker")
    es.add_argument("--audio")
    es.add_argument("--role", default="salesperson")
    es.set_defaults(fn=cmd_enroll_speaker)

    i = sub.add_parser("identify")
    i.add_argument("transcript")
    i.add_argument("--audio")
    i.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    i.set_defaults(fn=cmd_identify)

    sub.add_parser("list").set_defaults(fn=cmd_list)
    d = sub.add_parser("delete")
    d.add_argument("name")
    d.set_defaults(fn=cmd_delete)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
