import os
import shutil
import tempfile
import threading
import time
import traceback
import uuid

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from forensics import analyze_image
from video_forensics import analyze_video

app = FastAPI(title="PixelSleuth Image & Video Forensics")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index.html")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".mpg", ".mpeg"}

MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB
JOB_TTL_SECONDS = 1800

# Analysis runs in a worker thread and reports progress into this dict, which the
# frontend polls. That is what lets the scanning screen show the real stage the
# backend has reached instead of a timer that guesses.
_jobs = {}
_jobs_lock = threading.Lock()


def _new_job():
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "percent": 0,
            "message": "Starting...",
            "state": "running",
            "result": None,
            "error": None,
            "created": time.time(),
        }
    return job_id


def _update_job(job_id, **fields):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job.update(fields)


def _purge_old_jobs():
    cutoff = time.time() - JOB_TTL_SECONDS
    with _jobs_lock:
        for jid in [j for j, v in _jobs.items() if v["created"] < cutoff]:
            _jobs.pop(jid, None)


def _run_job(job_id, func, path):
    """Worker body: run the analysis, then always clean up the uploaded file."""
    def progress(percent, message):
        _update_job(job_id, percent=int(percent), message=message)

    try:
        result = func(path, progress=progress)
        _update_job(job_id, state="done", percent=100, message="Done", result=result)
    except Exception as exc:
        traceback.print_exc()
        _update_job(job_id, state="error", message="Analysis failed",
                    error=_friendly_error(exc))
    finally:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def _friendly_error(exc):
    text = str(exc)
    if "cannot identify image file" in text:
        return "That file could not be read as an image. Please upload a JPG, PNG or WEBP."
    if "could not be opened" in text or "No frames" in text:
        return text
    return f"The file could not be analyzed. ({type(exc).__name__})"


async def _save_upload(file, allowed_exts, kind):
    """Save the upload to a temp file, rejecting anything we cannot analyze."""
    name = file.filename or ""
    ext = os.path.splitext(name)[1].lower()
    if ext not in allowed_exts:
        return None, (f"'{name or 'this file'}' is not a supported {kind}. "
                      f"Supported: {', '.join(sorted(e[1:] for e in allowed_exts))}.")

    fd, path = tempfile.mkstemp(suffix=ext)
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    out.close()
                    os.remove(path)
                    return None, "That file is larger than the 200 MB limit."
                out.write(chunk)
    except Exception:
        if os.path.exists(path):
            os.remove(path)
        raise

    if size == 0:
        os.remove(path)
        return None, "That file is empty."
    return path, None


def _start(path, func):
    _purge_old_jobs()
    job_id = _new_job()
    threading.Thread(target=_run_job, args=(job_id, func, path), daemon=True).start()
    return JSONResponse({"job_id": job_id})


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    if not os.path.exists(INDEX_PATH):
        return HTMLResponse(
            content="<h2>Error: index.html not found!</h2>"
                    "<p>Make sure index.html is in the same folder as app.py.</p>",
            status_code=404,
        )
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/analyze-image")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    path, error = await _save_upload(file, IMAGE_EXTS, "image")
    if error:
        return JSONResponse({"error": error}, status_code=400)
    return _start(path, analyze_image)


@app.post("/analyze-video")
async def analyze_video_endpoint(file: UploadFile = File(...)):
    path, error = await _save_upload(file, VIDEO_EXTS, "video")
    if error:
        return JSONResponse({"error": error}, status_code=400)
    return _start(path, analyze_video)


@app.get("/progress/{job_id}")
async def progress_endpoint(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return JSONResponse({"error": "Unknown or expired job."}, status_code=404)
        return JSONResponse({
            "percent": job["percent"],
            "message": job["message"],
            "state": job["state"],
            "error": job["error"],
        })


@app.get("/result/{job_id}")
async def result_endpoint(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return JSONResponse({"error": "Unknown or expired job."}, status_code=404)
        if job["state"] == "error":
            return JSONResponse({"error": job["error"]}, status_code=500)
        if job["state"] != "done":
            return JSONResponse({"error": "Analysis is still running."}, status_code=409)
        return JSONResponse(job["result"])


@app.post("/analyze")
async def analyze_legacy(file: UploadFile = File(...)):
    """Original synchronous route, kept so existing callers do not break.
    Returns the same JSON as /analyze-image, just without progress reporting."""
    path, error = await _save_upload(file, IMAGE_EXTS, "image")
    if error:
        return JSONResponse({"error": error}, status_code=400)
    try:
        return JSONResponse(analyze_image(path))
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse({"error": _friendly_error(exc)}, status_code=400)
    finally:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
