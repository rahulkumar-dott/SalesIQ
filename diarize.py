"""Detect speakers with sherpa-onnx and merge them into a transcript JSON.

Usage:
    python diarize.py data/audio/sample-1.mp3 output/sample-1.json [--speakers 2]

Reads the transcript written by transcribe.py, adds a `speaker` to every
segment/word, and writes <transcript>.speakers.json.
No Hugging Face token needed; models live in ./models.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import sherpa_onnx

ROOT = Path(__file__).parent
SEG_MODEL = ROOT / "models/sherpa-seg/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMB_MODEL = ROOT / "models/sherpa-emb/nemo_en_titanet_small.onnx"


def load_audio(path: str, sr: int) -> np.ndarray:
    cmd = ["ffmpeg", "-nostdin", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def diarize(audio_path: str, num_speakers: int = -1, threshold: float = 0.8, min_seconds: float = 10.0):
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(SEG_MODEL)),
            num_threads=4,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMB_MODEL), num_threads=4),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers, threshold=threshold),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        sys.exit("invalid sherpa-onnx config (check model paths)")
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    samples = load_audio(audio_path, sd.sample_rate)
    result = sd.process(samples).sort_by_start_time()
    turns = [(r.start, r.end, f"SPEAKER_{r.speaker:02d}") for r in result]
    return merge_tiny_speakers(turns, min_seconds)


def merge_tiny_speakers(turns, min_seconds: float):
    """Fold speakers with < min_seconds of speech into the nearest real speaker."""
    totals = {}
    for s, e, k in turns:
        totals[k] = totals.get(k, 0.0) + (e - s)
    real = [t for t in turns if totals[t[2]] >= min_seconds]
    if not real or len(real) == len(turns):
        return turns
    out = []
    for s, e, k in turns:
        if totals[k] < min_seconds:
            mid = (s + e) / 2
            k = min(real, key=lambda t: min(abs(t[0] - mid), abs(t[1] - mid)))[2]
        out.append((s, e, k))
    # renumber so labels are contiguous: SPEAKER_00, SPEAKER_01, ...
    names = {k: f"SPEAKER_{i:02d}" for i, k in enumerate(sorted({t[2] for t in out}))}
    return [(s, e, names[k]) for s, e, k in out]


def speaker_at(turns, start: float, end: float):
    """Speaker whose turns overlap [start, end] the most."""
    best, best_ov = None, 0.0
    for s, e, spk in turns:
        ov = min(e, end) - max(s, start)
        if ov > best_ov:
            best, best_ov = spk, ov
    if best is None:  # no overlap: take the nearest turn
        mid = (start + end) / 2
        best = min(turns, key=lambda t: min(abs(t[0] - mid), abs(t[1] - mid)))[2]
    return best


def assign_words(segments, turns):
    """Give every word a speaker and a start time that respects speaker turns.

    Whisper's word *ends* are reliable but its *starts* drift early, sometimes by
    seconds when a word follows a pause. Sherpa's turn boundaries are accurate, so:
      - the speaker is the turn containing the word's end (nearest turn if in a gap)
      - the word's start is clamped so it never begins before that turn does
    """
    for seg in segments:
        for w in seg.get("words", []):
            if "start" not in w or "end" not in w:
                continue
            end = w["end"] - 0.02
            turn = next((t for t in turns if t[0] <= end <= t[1]), None)
            if turn is None:
                mid = (w["start"] + w["end"]) / 2
                turn = min(turns, key=lambda t: min(abs(t[0] - mid), abs(t[1] - mid)))
            elif w["start"] < turn[0] < w["end"]:
                w["start"] = round(turn[0], 3)
            w["speaker"] = turn[2]
    fix_stranded_words(segments, turns)
    for seg in segments:
        words = [w for w in seg.get("words", []) if "speaker" in w]
        if words:
            seg["start"], seg["end"] = words[0]["start"], words[-1]["end"]
            dur = {}
            for w in words:
                dur[w["speaker"]] = dur.get(w["speaker"], 0.0) + (w["end"] - w["start"])
            seg["speaker"] = max(dur, key=dur.get)


def speech_fraction(turns, start: float, end: float) -> float:
    """Share of [start, end] that falls inside any detected speech turn."""
    dur = max(end - start, 1e-6)
    return sum(max(0.0, min(e, end) - max(s, start)) for s, e, _ in turns) / dur


def fix_stranded_words(segments, turns, min_gap: float = 1.5, max_chars: int = 6,
                       word_len: float = 0.3, max_speech: float = 0.5):
    """Move short words that Whisper stranded on the wrong side of a silence.

    Pattern: a short word glued to the previous word (no gap before), followed by a
    long silence, with the next word spoken by someone else. Whisper times the first
    word of a new turn badly when it follows silence, attaching it to the previous
    speech. Such a word really belongs just before the next word.

    The word must also sit mostly in silence (little overlap with detected speech):
    a genuine last word of a turn lies inside speech and must stay where it is.
    """
    words = [w for s in segments for w in s.get("words", []) if "speaker" in w]
    for i in range(1, len(words) - 1):
        prev, w, nxt = words[i - 1], words[i], words[i + 1]
        short = len(re.sub(r"\W", "", w["word"])) <= max_chars
        glued = w["start"] - prev["end"] < 0.05
        silence_after = nxt["start"] - w["end"] >= min_gap
        in_silence = speech_fraction(turns, w["start"], w["end"]) < max_speech
        if short and glued and silence_after and in_silence and nxt["speaker"] != w["speaker"]:
            w["end"] = round(nxt["start"] - 0.02, 3)
            w["start"] = round(w["end"] - word_len, 3)
            w["speaker"] = nxt["speaker"]
            w["moved"] = True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("transcript")
    p.add_argument("--speakers", type=int, default=-1, help="exact count; -1 = auto")
    p.add_argument("--threshold", type=float, default=0.8, help="auto mode: lower = more speakers")
    p.add_argument("--min-seconds", type=float, default=10.0, help="speakers with less speech are merged away")
    a = p.parse_args()

    turns = diarize(a.audio, a.speakers, a.threshold, a.min_seconds)
    data = json.loads(Path(a.transcript).read_text(encoding="utf-8"))

    assign_words(data["segments"], turns)

    speakers = sorted({t[2] for t in turns})
    data["num_speakers"] = len(speakers)
    data["speakers"] = speakers
    data["speaker_turns"] = [{"start": round(s, 3), "end": round(e, 3), "speaker": k} for s, e, k in turns]

    out = Path(a.transcript).with_suffix(".speakers.json")
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"speakers detected: {len(speakers)} {speakers}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
