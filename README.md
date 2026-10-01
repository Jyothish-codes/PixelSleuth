# 🔍 PixelSleuth — Image & Video Forensics

PixelSleuth analyses an uploaded image or video and reports, in plain English, whether it shows
signs of AI generation or editing. It runs several independent forensic checks, marks the specific
areas that measured differently, and explains each one in a single sentence.

It is built to be honest about uncertainty: every result says what *could not* be determined, and
the interface never claims proof.

---

## What it does

**For images**

- Runs six forensic checks and translates each into plain language
- Gives one overall verdict: **LOOKS REAL**, **NOT SURE**, **MAYBE EDITED** or **LIKELY AI**
- Places numbered markers on the exact regions that measured differently, each with a
  one-sentence explanation shown on the marker itself
- Keeps the raw technical output in a collapsible **View technical details** section

**For videos**

- Samples frames evenly across the clip and runs the image checks on each
- Shows a timeline with every sampled frame colour-coded normal / suspicious / unclear
- Clicking a frame opens it with its own markers and explanations
- Summarises which timestamps looked unusual

---

## The forensic checks

| Check | What it looks at |
|---|---|
| **Metadata** | EXIF camera information, and software signatures such as Photoshop or Midjourney |
| **Authenticity Record** | C2PA provenance manifest, including AI-generation declarations |
| **Pixel Analysis** | Error Level Analysis — how different areas respond to re-compression |
| **Noise Pattern** | Local noise residual consistency across the image |
| **Lighting** | Whether brightness follows a natural gradient |
| **AI Detection** | Neural classifiers trained to recognise generated imagery |

Checks that cannot run are reported as such rather than counted against the image. A PNG screenshot
has no EXIF, no C2PA record and no JPEG history, so three checks are silent for reasons unrelated to
whether the picture is genuine — the result says so explicitly.

---

## Tech stack

- **Python** · **FastAPI** · **Uvicorn**
- **OpenCV** and **NumPy** for the pixel-level analysis
- **Pillow** for image and EXIF handling
- **PyTorch** + **Hugging Face Transformers** for AI detection
- **c2pa-python** for provenance manifests
- **Google Gemini** (optional) for a written explanation of the findings
- Frontend is plain HTML/CSS/JavaScript — no framework

---

## Requirements

- **Python 3.11+** (developed and tested on 3.14)
- **~2 GB free RAM** — two vision models stay loaded in memory
- **~1.1 GB disk** for model weights, downloaded automatically on first run

---

## Setup

```bash
git clone https://github.com/Jyothish-codes/PixelSleuth.git
cd PixelSleuth
pip install -r requirements.txt
```

### Optional: Gemini explanation

The app works fully without this. If no key is set it falls back to a plain-text summary instead of
an AI-written one.

```bash
cp .env.example .env
```

Then open `.env` and paste your key:

```
GEMINI_API_KEY="your_key_here"
```

Get one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). A Gemini API key begins
with `AIza` — an OAuth token starting with `AQ.` will be rejected by the API.

`.env` is gitignored, so the key never reaches version control.

---

## Running

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000**

The first start takes a minute or two while the models download. Later starts take a few seconds.

---

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | Serves the web interface |
| `POST` | `/analyze-image` | Upload an image; returns `{"job_id": "..."}` |
| `POST` | `/analyze-video` | Upload a video; returns `{"job_id": "..."}` |
| `GET` | `/progress/{job_id}` | Live percentage and current stage |
| `GET` | `/result/{job_id}` | Final result once the job is done |
| `POST` | `/analyze` | Original synchronous image route, kept for compatibility |

Analysis runs in a worker thread and reports real progress, so the scanning screen reflects the
stage the backend has actually reached rather than a timer.

Supported uploads: `jpg` `jpeg` `png` `webp` `bmp` `tif` `tiff` · `mp4` `mov` `avi` `mkv` `webm`
`m4v` `wmv` `mpg` `mpeg` — up to 200 MB.

### Example response

```json
{
  "media_type": "image",
  "overall_result": "MAYBE EDITED",
  "overall_level": "warning",
  "summary": "Parts of this image look like they may have been changed.",
  "checks": [
    {
      "key": "noise",
      "label": "Noise Pattern",
      "status": "suspicious",
      "message": "Some areas do not match the rest of the image."
    }
  ],
  "suspicious_points": [
    {
      "id": 1,
      "x": 373, "y": 280, "width": 160, "height": 120,
      "reason": "Unusual pixel pattern detected here."
    }
  ]
}
```

---

## Project structure

```
app.py              FastAPI routes, upload validation, background job tracking
forensics.py        The six image checks, region extraction, plain-English layer
video_forensics.py  Frame sampling and per-frame analysis
index.html          Complete frontend (styles and script inline)
requirements.txt    Python dependencies
.env.example        Template for the optional Gemini key
```

---

## Limitations

These are real and worth reading before trusting a result.

**AI detection is not solved.** The primary detector is
[`haywoodsloan/ai-image-detector-deploy`](https://huggingface.co/haywoodsloan/ai-image-detector-deploy).
It replaced an older pair of models that were measured getting modern photorealistic generations
confidently wrong. On the small sample used during development the new detector was correct on all
of it — but that sample was **five images**, which is far too few to quote as an accuracy figure. No
detector generalises to every generator, and new models appear constantly.

A second detector (`Organika/sdxl-detector`) is still consulted and reported, but it no longer
influences the verdict: on the same sample it was wrong in both directions, missing an obvious
generated image and flagging a real photo as artificial.

**Error Level Analysis reacts to focus, not just editing.** A photo with a sharp subject against a
blurred background will flag areas around the subject. When the other checks agree the picture is
ordinary, these are presented as *areas that looked different* with a matching explanation, rather
than as suspected edits — but the underlying check still reports them.

**Pixel analysis is weaker on video.** A frame decoded from a compressed stream carries no
independent JPEG history, so the same measurement reads as a texture difference rather than evidence
of a past edit. Video frames are therefore judged at a stricter cutoff, and the technical notes say
so.

**Absence of evidence is not evidence.** Most images on the internet have no EXIF and no C2PA
record. That is normal and is never treated as suspicious.

**Lighting analysis is weak alone.** Real scenes with several light sources break a simple linear
gradient model, so lighting never places a marker on its own — it needs corroboration from a
stronger check.

> Forensic signals indicate likelihood, not proof. Treat any result as one piece of evidence, not a
> final answer.

---

## License

No license specified — ask the repository owner before reuse.
