"""Background analysis jobs for the API: pick a dataset sample or upload a file,
then run transcribe.py -> segment.py as subprocesses, one job at a time (CPU bound).

Jobs live in memory; results land in output/ like a command-line run.
"""
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "audio"
UPLOADS = ROOT / "data" / "uploads"
OUT = ROOT / "output"
DATASET = "liva-ai/mock-sales"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/main/data/{{}}.mp3"
SAMPLES = [f"sample-{i}" for i in range(1, 11)]
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma", ".amr", ".aiff", ".aif"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpeg", ".mpg", ".wmv", ".3gp", ".flv", ".ts", ".mts"}
ALLOWED = AUDIO_EXT | VIDEO_EXT
MAX_UPLOAD = 1024 * 1024 * 1024  # 1 GB: videos are big. Streamed to disk, never held in memory.
MAX_MINUTES = float(os.environ.get("MAX_AUDIO_MINUTES") or 60)  # CPU transcription time grows with length

JOBS: dict = {}
_q: "queue.Queue[str]" = queue.Queue()
_started = False


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "call"


def dataset_samples() -> list:
    accents = {}
    ref = ROOT / "data" / "reference" / "train.jsonl"
    if ref.exists():
        for line in ref.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            accents[Path(d["audio"]).stem] = [d.get("speaker_1_accent"), d.get("speaker_2_accent")]
    return [
        {
            "name": s,
            "downloaded": (DATA / f"{s}.mp3").exists(),
            "analysed": any((OUT / f"{s}{x}.utterances.json").exists() for x in ("", ".speakers")),
            "accents": accents.get(s, []),
        }
        for s in SAMPLES
    ]


def _new_job(name: str, audio: Path, source: str) -> dict:
    job = {
        "id": uuid.uuid4().hex[:8], "name": name, "source": source, "audio": str(audio),
        "status": "queued", "step": "queued", "log": [], "error": None, "created": time.time(),
    }
    JOBS[job["id"]] = job
    return job


def submit(job: dict) -> dict:
    global _started
    if not _started:
        threading.Thread(target=_worker, daemon=True).start()
        _started = True
    _q.put(job["id"])
    return job


def submit_dataset(sample: str) -> dict:
    if sample not in SAMPLES:  # whitelist: nothing user-supplied reaches a path or URL
        raise ValueError(f"unknown sample {sample!r}")
    return submit(_new_job(sample, DATA / f"{sample}.mp3", "dataset"))


def _tool(name: str) -> str:
    exe = os.environ.get(f"{name.upper()}_PATH") or shutil.which(name)
    if not exe:
        raise ValueError(f"{name} is not installed or not on PATH (needed to read audio/video files)")
    return exe


def probe(path: Path) -> dict:
    """{has_audio, duration_s} from ffprobe; raises ValueError if the file is not readable media."""
    try:
        out = subprocess.run(
            [_tool("ffprobe"), "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=codec_name:format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout
    except subprocess.TimeoutExpired:
        raise ValueError("could not read the file (timed out)")
    except subprocess.CalledProcessError:
        raise ValueError("this file is not a valid audio or video file")
    info = json.loads(out or "{}")
    try:
        duration = float((info.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        duration = 0.0
    return {"has_audio": bool(info.get("streams")), "duration_s": duration}


def extract_audio(src: Path, dst: Path) -> None:
    """Decode only the audio stream to 16 kHz mono WAV (-vn: any video stream is ignored)."""
    tmp = dst.with_suffix(".part.wav")
    r = subprocess.run(
        [_tool("ffmpeg"), "-nostdin", "-y", "-i", str(src), "-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(tmp)],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise RuntimeError("audio extraction failed: " + (r.stderr or "").strip().splitlines()[-1][:200])
    tmp.replace(dst)


def submit_upload(filename: str, tmp_path: Path) -> dict:
    """Validate an already-saved upload (tmp_path) and queue it. The caller deletes tmp_path on ValueError."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED:
        raise ValueError(f"unsupported file type {ext or '(none)'}; send an audio or video file "
                         f"({', '.join(sorted(AUDIO_EXT | VIDEO_EXT))})")
    info = probe(tmp_path)  # the extension is not trusted: ffprobe must find a real audio stream
    if not info["has_audio"]:
        raise ValueError("this file has no audio track" + (" (video only)" if ext in VIDEO_EXT else ""))
    if info["duration_s"] / 60 > MAX_MINUTES:
        raise ValueError(f"too long: {info['duration_s'] / 60:.0f} min (limit {MAX_MINUTES:.0f} min; set MAX_AUDIO_MINUTES)")
    stem = slug(Path(filename).stem)
    name, n = stem, 2
    while (UPLOADS / f"{name}.wav").exists() or any(UPLOADS.glob(f"{name}.src.*")) or (OUT / f"{name}.json").exists():
        name = f"{stem}-{n}"
        n += 1
    UPLOADS.mkdir(parents=True, exist_ok=True)
    src = UPLOADS / f"{name}.src{ext}"
    tmp_path.replace(src)
    job = _new_job(name, UPLOADS / f"{name}.wav", "upload")
    job["media"] = str(src)
    job["kind"] = "video" if ext in VIDEO_EXT else "audio"
    job["duration_s"] = round(info["duration_s"], 1)
    return submit(job)


def _log(job: dict, line: str) -> None:
    job["log"].append(line.rstrip())
    del job["log"][:-60]


def _run(job: dict, args: list, step: str) -> None:
    job["step"] = step
    p = subprocess.Popen(
        [sys.executable, *args], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    for line in p.stdout:
        _log(job, line)
    if p.wait() != 0:
        tail = " | ".join(job["log"][-3:])
        raise RuntimeError(f"{step} failed: {tail[:400]}")


def _worker() -> None:
    while True:
        job = JOBS[_q.get()]
        job["status"] = "running"
        try:
            audio = Path(job["audio"])
            if job["source"] == "upload" and not audio.exists():
                job["step"] = "extracting audio"
                src = Path(job["media"])
                extract_audio(src, audio)
                src.unlink(missing_ok=True)  # only the audio is kept; the original (and any video) is deleted
            if job["source"] == "dataset" and not audio.exists():
                job["step"] = "downloading"
                DATA.mkdir(parents=True, exist_ok=True)
                tmp = audio.with_suffix(".part")
                urllib.request.urlretrieve(DATASET_URL.format(job["name"]), tmp)
                tmp.replace(audio)
            _run(job, ["-u", "transcribe.py", str(audio)], "transcribing")
            _run(job, ["-u", "segment.py", str(OUT / f"{audio.stem}.json"), "--roles", "auto"], "analysing")
            job["status"], job["step"] = "done", "done"
        except Exception as e:  # noqa: BLE001 - surface any failure to the UI
            job["status"], job["step"], job["error"] = "failed", "failed", str(e)


def public(job: dict) -> dict:
    return {k: job[k] for k in ("id", "name", "source", "status", "step", "error", "created")} | {
        "log": job["log"][-8:],
        "kind": job.get("kind"),
        "duration_s": job.get("duration_s"),
    }
