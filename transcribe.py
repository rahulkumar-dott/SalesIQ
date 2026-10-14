"""Transcribe a call with word timestamps and speaker labels.

Whisper (faster-whisper) for text + word timestamps, sherpa-onnx for speakers.

Usage:
    python transcribe.py data/audio/sample-1.mp3 [--model medium.en] [--speakers 2]

Writes output/<audio>.json and prints a readable transcript.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from faster_whisper import WhisperModel

import config  # noqa: F401  (loads .env: WHISPER_MODEL, WHISPER_LANGUAGE)
import diarize as dz
import pii


def fmt(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def transcribe(audio: str, model_name: str, device: str, compute_type: str, prompt: str | None = None,
               language: str | None = None):
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        audio, language=language, word_timestamps=True, vad_filter=True, beam_size=5,  # None = auto-detect
        initial_prompt=prompt or None,  # a punctuated sample steers Whisper's output style
    )
    out = []
    for s in segments:  # generator: decoding happens as we iterate
        out.append(
            {
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": s.text.strip(),
                "words": [
                    {"word": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3),
                     "score": round(w.probability, 3)}
                    for w in (s.words or [])
                ],
            }
        )
        print(f"  ... {fmt(s.end)} / {fmt(info.duration)}", flush=True)
    return out, info.duration, {"language": info.language, "probability": round(float(info.language_probability), 3)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--model", default=os.environ.get("WHISPER_MODEL", "medium.en"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--compute-type", default="int8")
    p.add_argument("--speakers", type=int, default=-1, help="exact speaker count; -1 = auto")
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--out", default="output")
    p.add_argument("--language", default=os.environ.get("WHISPER_LANGUAGE", "auto"),
                   help="auto | en | hi | ... (ignored for English-only *.en models)")
    p.add_argument("--prompt", default=None,
                   help="initial prompt written in normal punctuated style, to make Whisper punctuate")
    a = p.parse_args()

    print(f"[1/3] transcribing with {a.model} ({a.device}, {a.compute_type})", flush=True)
    # English-only models (*.en) can only do English. Multilingual models auto-detect unless --language is given.
    lang = a.language if a.language not in (None, "", "auto") else ("en" if a.model.endswith(".en") else None)
    segments, duration, lang_info = transcribe(a.audio, a.model, a.device, a.compute_type, a.prompt, lang)
    print(f"      language: {lang_info['language']} (p={lang_info['probability']})", flush=True)

    print("[2/3] detecting speakers", flush=True)
    turns = dz.diarize(a.audio, a.speakers, a.threshold)
    dz.assign_words(segments, turns)
    if pii.enabled() and os.environ.get("PII_STORE_RAW") != "1":
        found = pii.scrub_segments(segments)  # personal identifiers never reach the stored transcript
        print(f"      PII masked in stored transcript: {dict(found) or 'none found'}", flush=True)
    speakers = sorted({t[2] for t in turns})

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (Path(a.audio).stem + ".json")
    out.write_text(
        json.dumps(
            {
                "audio": a.audio,
                "duration": round(duration, 2),
                "model": a.model,
                "language": lang_info["language"],
                "language_probability": lang_info["probability"],
                "num_speakers": len(speakers),
                "speakers": speakers,
                "speaker_turns": [
                    {"start": round(s, 3), "end": round(e, 3), "speaker": k} for s, e, k in turns
                ],
                "segments": segments,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[3/3] saved {out}", flush=True)
    print(f"\nspeakers detected: {len(speakers)} {speakers}\n")
    for s in segments:
        print(f"[{fmt(s['start'])}] {s['speaker']}: {s['text']}")


if __name__ == "__main__":
    sys.exit(main())
