"""Read-only HTTP API over the analysed calls in output/.

Run:
    .venv\\Scripts\\python.exe api.py            # http://127.0.0.1:8000  (docs at /docs)

Endpoints:
    GET /health
    GET /calls                       list analysed calls (outcome, summary, headline metrics)
    GET /calls/{name}                everything: roles, insights, metrics, summary, utterances
    GET /calls/{name}/metrics        audio + sales metrics
    GET /calls/{name}/insights       outcome, objections, buying signals, rep performance, ...
    GET /calls/{name}/utterances     speaker-labelled segments with intent
    GET /aggregate                   win rate, objections, stages, per-rep scorecards
    GET /budget                      LLM spend so far
    GET /dataset                     the 10 dataset samples (downloaded / analysed flags)
    POST /jobs/dataset {sample}      analyse a dataset sample (downloads it if needed)
    POST /jobs/upload  (file)        analyse an uploaded audio or video file (audio track only)
    GET /jobs, /jobs/{id}            job status (queued/running/done/failed/cancelled) and log
    POST /jobs/{id}/cancel           cancel a queued or running job

Analysis is produced by transcribe.py -> segment.py (run from the CLI, or as a job via
POST /jobs/*); the GET endpoints only read the results. If ui/dist exists it is served at /.
There is no authentication and it binds to localhost: do not expose it as is, the
transcripts are customer data.
"""
import base64
import json
import os
import secrets
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import aggregate
import budget
import config  # noqa: F401  (loads .env so SALESIQ_PASSWORD is seen)
import jobs

OUT = Path(__file__).parent / "output"
SUFFIX = ".speakers.utterances.json"

app = FastAPI(title="SalesIQ", description="Sales-call metrics and insights", version="0.1")


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    """HTTP Basic auth on everything when SALESIQ_PASSWORD is set (required before sharing publicly)."""
    password = os.environ.get("SALESIQ_PASSWORD")
    if password:
        user = os.environ.get("SALESIQ_USER", "salesiq")
        ok = False
        header = request.headers.get("authorization", "")
        if header.lower().startswith("basic "):
            try:
                u, _, p = base64.b64decode(header[6:]).decode("utf-8").partition(":")
                ok = secrets.compare_digest(u, user) & secrets.compare_digest(p, password)
            except (ValueError, UnicodeDecodeError):
                ok = False
        if not ok:
            return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="SalesIQ"'})
    return await call_next(request)


def _files() -> dict:
    """name -> newest result file (a name may have both .speakers.utterances.json and .utterances.json)."""
    found = {}
    for f in OUT.glob("*.utterances.json"):
        name = f.name[: -len(SUFFIX)] if f.name.endswith(SUFFIX) else f.name[: -len(".utterances.json")]
        if name not in found or f.stat().st_mtime > found[name].stat().st_mtime:
            found[name] = f
    return found


def _names() -> list:
    return sorted(_files())


def _load(name: str) -> dict:
    # only exact names that exist on disk are served, so no path traversal is possible
    files = _files()
    if name not in files:
        raise HTTPException(404, f"no analysed call {name!r}; have {sorted(files)}")
    return json.loads(files[name].read_text(encoding="utf-8"))


def _summary(name: str, d: dict) -> dict:
    ins, m = d.get("insights") or {}, d.get("metrics") or {}
    a, s = m.get("audio") or {}, m.get("sales") or {}
    return {
        "name": name,
        "roles": d.get("roles"),
        "outcome": ins.get("outcome"),
        "conversation": d.get("conversation"),
        "category": d.get("category"),
        "category_label": d.get("category_label"),
        "product_sold": s.get("product_sold"),
        "deal": s.get("deal") or ({"amount": s["deal_value_usd"], "currency": "USD"} if s.get("deal_value_usd") else None),
        "stage_reached": s.get("stage_reached"),
        "pitch_attempts": s.get("pitch_attempts"),
        "close_attempts": s.get("close_attempts"),
        "duration_s": a.get("duration_s"),
        "talk_ratio": a.get("talk_ratio"),
        "summary": d.get("summary"),
    }


@app.get("/health")
def health():
    return {"status": "ok", "calls": len(_names())}


@app.get("/calls")
def calls():
    return [_summary(n, _load(n)) for n in _names()]


@app.get("/calls/{name}")
def call(name: str):
    d = _load(name)
    return {
        "name": name,
        "audio": d.get("audio"),
        "num_speakers": d.get("num_speakers"),
        "role_method": d.get("role_method"),
        "roles": d.get("roles"),
        "role_evidence": d.get("role_evidence"),
        "summary": d.get("summary"),
        "insights": d.get("insights"),
        "conversation": d.get("conversation"),
        "consistency": d.get("consistency"),
        "category": d.get("category"),
        "category_label": d.get("category_label"),
        "details": d.get("details"),
        "qualification": d.get("qualification"),
        "compliance": d.get("compliance"),
        "metrics": d.get("metrics"),
        "utterances": d.get("utterances"),
    }


@app.get("/calls/{name}/metrics")
def call_metrics(name: str):
    return _load(name).get("metrics") or {}


@app.get("/calls/{name}/insights")
def call_insights(name: str):
    d = _load(name)
    return {
        "summary": d.get("summary"),
        "insights": d.get("insights"),
        "roles": d.get("roles"),
        "category": d.get("category"),
        "details": d.get("details"),
        "qualification": d.get("qualification"),
        "compliance": d.get("compliance"),
    }


@app.get("/calls/{name}/utterances")
def call_utterances(name: str):
    return _load(name).get("utterances") or []


@app.get("/aggregate")
def agg():
    try:
        return aggregate.build(OUT)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/dataset")
def dataset():
    return {"dataset": jobs.DATASET, "source": "Hugging Face", "samples": jobs.dataset_samples()}


class DatasetJob(BaseModel):
    sample: str


@app.post("/jobs/dataset", status_code=202)
def job_dataset(body: DatasetJob):
    try:
        return jobs.public(jobs.submit_dataset(body.sample))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/jobs/upload", status_code=202)
async def job_upload(file: UploadFile = File(...)):
    """Any audio or video file. Streamed to disk (videos are large); only the audio track is used."""
    jobs.UPLOADS.mkdir(parents=True, exist_ok=True)
    tmp = jobs.UPLOADS / f".incoming-{secrets.token_hex(8)}"
    try:
        size = 0
        with open(tmp, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > jobs.MAX_UPLOAD:
                    raise HTTPException(413, f"file too large ({jobs.MAX_UPLOAD // (1024 * 1024)} MB max)")
                out.write(chunk)
        return jobs.public(jobs.submit_upload(file.filename or "media", tmp))
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        tmp.unlink(missing_ok=True)  # already moved on success


@app.get("/jobs")
def list_jobs():
    return [jobs.public(j) for j in sorted(jobs.JOBS.values(), key=lambda j: -j["created"])]


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs.JOBS:
        raise HTTPException(404, "no such job")
    return jobs.public(jobs.JOBS[job_id], full=True)


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if job_id not in jobs.JOBS:
        raise HTTPException(404, "no such job")
    return jobs.public(jobs.cancel(job_id), full=True)


@app.get("/budget")
def spend():
    s = budget.load()
    return {**s, "limit_usd": budget.limit(), "remaining_usd": round(budget.limit() - s["usd"], 6)}


UI = Path(__file__).parent / "ui" / "dist"
if UI.exists():  # production: `npm run build` in ui/, then python api.py serves the UI too
    app.mount("/", StaticFiles(directory=UI, html=True), name="ui")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
