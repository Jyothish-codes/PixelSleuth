import os
import json
import base64
import numpy as np
import cv2
from PIL import Image
from PIL.ExifTags import TAGS
from transformers import pipeline
from google import genai

# Lazy loading for c2pa to handle missing installations gracefully
try:
    import c2pa
    HAS_C2PA = True
except ImportError:
    HAS_C2PA = False

# ============================================================
# >>> CHANGE HERE: your Gemini API key <<<
# Set it as an environment variable before running, do not paste it into this file:
#   Windows (cmd):        set GEMINI_API_KEY=your_key_here
#   Mac/Linux:             export GEMINI_API_KEY=your_key_here
# The code below reads it automatically. If GEMINI_API_KEY is not set, the app
# still works and falls back to a plain-text template report instead of crashing.
# ============================================================
GEMINI_MODEL = "gemini-2.5-flash"  # >>> CHANGE HERE if you want a different Gemini model <<<

# Load TWO independent Hugging Face models instead of one. A single dated model
# (dima806/ai_vs_real_image_detection) was found in testing to sit near 50/50 on
# out-of-training-domain images -- genuine uncertainty, not a bug to threshold away.
# Using two models that were trained differently means agreement is a real signal,
# and disagreement is honestly reported as inconclusive instead of faked.
print("Loading HuggingFace models (first run only, may take a minute)...")
hf_classifier_a = pipeline("image-classification", model="umm-maybe/AI-image-detector", device=-1)
hf_classifier_b = pipeline("image-classification", model="Organika/sdxl-detector", device=-1)


# ============================================================
# HELPERS
# ============================================================

def robust_z_score(data):
    median = np.median(data)
    mad = max(np.median(np.abs(data - median)), 1e-6)
    return (data - median) / (mad * 1.4826)


def get_grid_stats(img_array, grid_size, stat='mean'):
    grid_h, grid_w = grid_size
    h, w = img_array.shape[:2]
    ph, pw = h // grid_h, w // grid_w
    stats = np.zeros((grid_h, grid_w))
    for i in range(grid_h):
        for j in range(grid_w):
            patch = img_array[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw]
            stats[i, j] = np.mean(patch) if stat == 'mean' else np.std(patch)
    return stats


# ============================================================
# LAYER 1: FILE LEVEL
# ============================================================

def check_metadata(img):
    exif_data = img.getexif()
    if not exif_data:
        return {"name": "Metadata Check", "status": "removed",
                "detail": "No EXIF data found. This is common on social media and not proof of tampering."}

    suspicious_keywords = ["stable diffusion", "midjourney", "dall-e", "firefly", "imagen", "comfyui",
                            "photoshop", "gimp", "lightroom", "snapseed", "picsart", "canva", "facetune"]

    has_camera = False
    for tag_id, value in exif_data.items():
        tag_name = TAGS.get(tag_id, tag_id)
        val_str = str(value).lower()
        for kw in suspicious_keywords:
            if kw in val_str:
                return {"name": "Metadata Check", "status": "suspicious",
                        "detail": f"Suspicious software signature found: {kw}"}
        if tag_name in ["Make", "Model"] and val_str.strip():
            has_camera = True

    if has_camera:
        return {"name": "Metadata Check", "status": "ok", "detail": "Standard camera metadata found."}
    return {"name": "Metadata Check", "status": "inconclusive",
            "detail": "EXIF present but no specific camera or software signatures identified."}


def check_c2pa(file_path):
    if not HAS_C2PA:
        return {"name": "C2PA Check", "status": "unavailable",
                "detail": "C2PA library unavailable. Absence does not prove authenticity."}
    try:
        # FIX: c2pa.Reader.from_file(...) does not exist in this version of the
        # library (confirmed: raises AttributeError). The constructor takes the
        # path directly: c2pa.Reader(path).
        reader = c2pa.Reader(file_path)
        manifest_str = reader.json().lower()
        if "trainedalgorithmicmedia" in manifest_str:
            return {"name": "C2PA Check", "status": "suspicious",
                    "detail": "C2PA manifest declares trainedAlgorithmicMedia."}
        if "compositewithtrainedalgorithmicmedia" in manifest_str:
            return {"name": "C2PA Check", "status": "suspicious",
                    "detail": "C2PA manifest declares compositeWithTrainedAlgorithmicMedia."}
        return {"name": "C2PA Check", "status": "ok", "detail": "C2PA manifest found with no AI signatures."}
    except Exception:
        return {"name": "C2PA Check", "status": "unavailable",
                "detail": "No readable C2PA manifest found. Absence does not prove authenticity, "
                          "as most images lack C2PA data."}


# ============================================================
# LAYER 2: FORENSIC SIGNALS (local, patch-based)
# ============================================================

def check_ela(file_path, img_format):
    if img_format != 'JPEG':
        return {"name": "Error Level Analysis", "status": "inconclusive",
                "detail": "Skipped. Image is not a JPEG."}, None

    orig = cv2.imread(file_path)
    _, enc = cv2.imencode('.jpg', orig, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    recompressed = cv2.imdecode(enc, 1)
    diff = np.abs(orig.astype(np.float32) - recompressed.astype(np.float32)).mean(axis=2)

    grid_means = get_grid_stats(diff, (12, 12), 'mean')
    z_scores = robust_z_score(grid_means)
    extreme_count = int(np.sum(z_scores > 3.0))
    fraction = extreme_count / 144

    if 0 < fraction < 0.25:
        return {"name": "Error Level Analysis", "status": "suspicious",
                "detail": f"Localized anomaly detected ({extreme_count} patches)."}, z_scores
    elif fraction >= 0.25:
        return {"name": "Error Level Analysis", "status": "inconclusive",
                "detail": "Uniformly extreme patches. This usually means the whole image was "
                          "re-compressed (e.g., WhatsApp), not locally edited."}, z_scores
    return {"name": "Error Level Analysis", "status": "ok",
            "detail": "No significant localized compression differences."}, z_scores


def check_noise_residual(file_path):
    gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = cv2.absdiff(gray, blur)

    grid_std = get_grid_stats(noise, (12, 12), 'std')
    z_scores = robust_z_score(grid_std)
    extreme_count = int(np.sum(z_scores > 3.0))
    fraction = extreme_count / 144

    if 0 < fraction < 0.25:
        return {"name": "Noise Residual", "status": "suspicious",
                "detail": f"Localized noise anomaly detected ({extreme_count} patches)."}, z_scores
    elif fraction >= 0.25:
        return {"name": "Noise Residual", "status": "inconclusive",
                "detail": "Uniformly extreme noise variance. Global filtering applied, "
                          "localized editing undetectable."}, z_scores
    return {"name": "Noise Residual", "status": "ok",
            "detail": "Consistent noise signature across the image."}, z_scores


def check_lighting(file_path):
    img = cv2.imread(file_path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    channels = cv2.split(lab)
    flagged_patches = 0

    Y, X = np.indices((8, 8))
    A = np.c_[X.flatten(), Y.flatten(), np.ones(64)]

    for ch in channels:
        means = get_grid_stats(ch, (8, 8), 'mean')
        Z = means.flatten()
        C, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
        plane = (A @ C).reshape(8, 8)
        diff = np.abs(means - plane)
        flagged_patches += int(np.sum(robust_z_score(diff) > 3.0))

    fraction = flagged_patches / (64 * 3)
    if fraction > 0.03:
        return {"name": "Lighting Consistency", "status": "inconclusive",
                "detail": "Deviations detected. However, real scenes with multiple light sources "
                          "also break a simple linear gradient, so this signal is weak on its own."}
    return {"name": "Lighting Consistency", "status": "ok",
            "detail": "Lighting gradients follow natural linear constraints."}


# ============================================================
# LAYER 3: LEARNED MODEL
# ============================================================

def _artificial_score(results):
    """Both models use label names like 'artificial'/'human' rather than 'fake'/'real'.
    Match by substring (confirmed against both model cards) rather than an exact key,
    since trusting an unverified exact label name is what caused the earlier C2PA bug."""
    for r in results:
        label = r['label'].lower()
        if 'artificial' in label or 'ai' in label or 'fake' in label:
            return float(r['score'])
    return None


def check_ai_classifier(img_pil):
    try:
        rgb_img = img_pil.convert('RGB')
        res_a = hf_classifier_a(rgb_img)
        res_b = hf_classifier_b(rgb_img)

        # Uncomment while testing, to see each model's raw output on your own
        # real/AI sample images before trusting the combined verdict:
        # print("RAW A:", res_a, "RAW B:", res_b)

        score_a = _artificial_score(res_a)
        score_b = _artificial_score(res_b)

        # Known, documented limitation of this entire model family (from the
        # maintainer's own discussion page): screenshots, memes, and other generic
        # digital images are frequently misclassified, regardless of whether the
        # underlying content is AI-generated or real. Flag that honestly rather
        # than claim false confidence.
        domain_note = (" Note: both models are tuned for stylised AI art rather than general "
                        "photos/screenshots, and their maintainers report unreliable results on "
                        "memes and screenshots specifically.")

        if score_a is None or score_b is None:
            return {"name": "AI Classifier", "status": "inconclusive",
                    "detail": "Could not read a confidence score from one of the two models." + domain_note}

        agree_artificial = score_a >= 0.6 and score_b >= 0.6
        agree_human = score_a < 0.4 and score_b < 0.4

        if agree_artificial:
            return {"name": "AI Classifier", "status": "suspicious",
                    "detail": f"Both independent models flag artificial generation patterns "
                              f"(model A: {score_a:.2f}, model B: {score_b:.2f})." + domain_note}
        if agree_human:
            return {"name": "AI Classifier", "status": "ok",
                    "detail": f"Both independent models are consistent with a human-made/real image "
                              f"(model A artificial-score: {score_a:.2f}, model B: {score_b:.2f})." + domain_note}
        return {"name": "AI Classifier", "status": "inconclusive",
                "detail": f"The two models disagree or are uncertain (model A: {score_a:.2f}, "
                          f"model B: {score_b:.2f}). Reported as inconclusive rather than guessing." + domain_note}
    except Exception as e:
        return {"name": "AI Classifier", "status": "inconclusive", "detail": f"Failed to classify image: {str(e)}"}


# ============================================================
# EVIDENCE VISUALIZATION
# ============================================================

def generate_heatmap(file_path, ela_z, noise_z):
    if ela_z is None and noise_z is None:
        return None

    base_z = np.zeros((12, 12))
    if ela_z is not None:
        base_z = np.maximum(base_z, ela_z)
    if noise_z is not None:
        base_z = np.maximum(base_z, noise_z)

    peak = np.max(base_z)
    extreme_frac = np.sum(base_z > 3.0) / 144
    if peak < 3.0 or extreme_frac > 0.25:
        return None  # too weak, or anomaly is global rather than localized

    orig = cv2.imread(file_path)
    H, W = orig.shape[:2]
    heatmap = np.clip((base_z / peak) * 255, 0, 255).astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_CUBIC)
    color_map = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 0.6, color_map, 0.4, 0)

    _, buffer = cv2.imencode('.png', overlay)
    return base64.b64encode(buffer).decode('utf-8')


# ============================================================
# EXPLANATION LAYER (Gemini)
# ============================================================

def generate_explanation(checks, file_path):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("No Gemini API key set (see GEMINI_API_KEY note at top of this file)")

        client = genai.Client(api_key=api_key)
        prompt = (
            "Explain these image forensics findings in under 150 words in plain language. "
            "Explicitly list what could NOT be concluded. Never claim certainty and never invent "
            "a finding that isn't in the measurements provided.\n\n"
            f"{json.dumps(checks, indent=2)}"
        )
        img_pil = Image.open(file_path)
        response = client.models.generate_content(model=GEMINI_MODEL, contents=[prompt, img_pil])
        return response.text, "gemini"
    except Exception:
        bullets = "\n".join([f"- {c['name']}: {c['status'].upper()} ({c['detail']})" for c in checks])
        text = f"Automated Summary:\n{bullets}\n\nNote: Certainty cannot be guaranteed. Authentic images may show inconclusive flags."
        return text, "template"


# ============================================================
# VERDICT (shared helper so there is exactly one source of truth)
# ============================================================

def build_verdict(checks):
    suspicious_count = sum(1 for c in checks if c["status"] == "suspicious")
    ok_count = sum(1 for c in checks if c["status"] == "ok")
    ai_check = next((c for c in checks if c["name"] == "AI Classifier"), None)
    ai_is_suspicious = bool(ai_check and ai_check["status"] == "suspicious")

    if ai_is_suspicious and suspicious_count >= 2:
        return ("LIKELY AI GENERATED / MANIPULATED", "danger",
                f"AI classifier detected artificial generation patterns alongside {suspicious_count - 1} additional forensic anomaly.")
    if ai_is_suspicious:
        return ("POSSIBLY AI GENERATED", "warning",
                "AI classifier detected artificial generation patterns. Secondary forensic signals are inconclusive.")
    if suspicious_count >= 2:
        return ("LIKELY MANIPULATED / EDITED", "warning",
                f"{suspicious_count} independent forensic checks detected anomalies (e.g., noise residual, ELA, EXIF).")
    if suspicious_count == 1:
        return ("WEAK ANOMALY DETECTED", "warning",
                "A single forensic check flagged an anomaly (often caused by standard web re-compression). Treat with caution.")
    if ok_count >= 3:
        return ("LIKELY AUTHENTIC", "success",
                "Forensic checks, camera metadata, and classifier patterns align with authentic imagery.")
    return ("INCONCLUSIVE / UNCERTAIN", "neutral",
            "Signals are mixed or stripped (common with social media screenshots). Absence of proof is not proof of authenticity.")


# ============================================================
# MAIN PIPELINE
# ============================================================

def analyze_image(file_path):
    img_pil = Image.open(file_path)
    img_format = img_pil.format

    checks = []
    checks.append(check_metadata(img_pil))
    checks.append(check_c2pa(file_path))

    ela_res, ela_z = check_ela(file_path, img_format)
    checks.append(ela_res)
    noise_res, noise_z = check_noise_residual(file_path)
    checks.append(noise_res)
    checks.append(check_lighting(file_path))
    checks.append(check_ai_classifier(img_pil))

    verdict_status, verdict_level, verdict_detail = build_verdict(checks)
    heatmap_b64 = generate_heatmap(file_path, ela_z, noise_z)
    report_text, report_source = generate_explanation(checks, file_path)

    return {
        "verdict_status": verdict_status,
        "verdict_level": verdict_level,
        "verdict_detail": verdict_detail,
        "checks": checks,
        "report": report_text,
        "report_source": report_source,
        "heatmap": heatmap_b64,
    }
