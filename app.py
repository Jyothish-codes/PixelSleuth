import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from forensics import analyze_image

app = FastAPI(title="TrustLens Image Forensics")

# Locate index.html dynamically in the same directory as app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index.html")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    if not os.path.exists(INDEX_PATH):
        return HTMLResponse(
            content="<h2>Error: index.html not found!</h2><p>Make sure index.html is located in the exact same folder as app.py.</p>",
            status_code=404
        )
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    fd, path = tempfile.mkstemp(suffix="." + file.filename.split(".")[-1])
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(await file.read())
        result = analyze_image(path)
        return JSONResponse(content=result)
    finally:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
