import os
import json
import base64
import numpy as np
import cv2
from PIL import Image
from PIL.ExifTags import TAGS
from transformers import pipeline
from google import genai

# Load GEMINI_API_KEY (and any other settings) from the .env file sitting next to
# this file, so the key never has to be pasted into the source. A real environment
# variable, if one is already set, always wins over the .env value.
_HERE = os.path.dirname(os.path.abspath(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_HERE, ".env"))
except ImportError:
    pass  # python-dotenv not installed; fall back to plain environment variables

# This machine runs antivirus software that intercepts HTTPS and re-signs it with
# its own root certificate. Windows trusts that root, but Python ships its own CA
# bundle, so every outbound HTTPS call (Gemini, HuggingFace) fails verification.
# ca_bundle.pem is the standard CA list plus that local root; pointing at it keeps
# certificate verification fully ON. Delete the file to go back to the default.
_CA_BUNDLE = os.path.join(_HERE, "ca_bundle.pem")
if os.path.exists(_CA_BUNDLE) and not os.environ.get("SSL_CERT_FILE"):
    os.environ["SSL_CERT_FILE"] = _CA_BUNDLE
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _CA_BUNDLE)

# Lazy loading for c2pa to handle missing installations gracefully
try:
    import c2pa
    HAS_C2PA = True
except ImportError:
    HAS_C2PA = False

# ============================================================
# >>> Your Gemini API key goes in the .env file, NOT in this file <<<
# Open .env (same folder as this file) and set:
#   GEMINI_API_KEY="your_key_here"
# .env is listed in .gitignore, so the key stays out of version control.
# A real environment variable, if set, takes priority over the .env value.
# If no key is set, the app still works and falls back to a plain-text template
# report instead of crashing.
# ============================================================
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")  # override via .env if needed

# Two independent models, with a clear primary.
#
# The previous primary (umm-maybe/AI-image-detector, 2022) was measured failing on
# modern photorealistic generations: on a Midjourney-style test photo it returned
# 83% "human", and Organika agreed at 98% "human". Both were confidently wrong, so
# no threshold change could have rescued that call.
#
# haywoodsloan/ai-image-detector-deploy was measured on the same images and got
# every one right (AI photo 99.8% artificial; two real photos 98%+ real), so it is
# now the primary. Organika is kept as a second opinion because it is specifically
# tuned for SDXL output, but it no longer gets to veto a confident primary -- that
# is what turned a correct AI detection into "inconclusive".
print("Loading HuggingFace models (first run only, may take a minute)...")
hf_classifier_a = pipeline("image-classification",
                           model="haywoodsloan/ai-image-detector-deploy", device=-1)
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

    # Keep the per-channel z-scores (not just the count) so the same measurement
    # can also drive the on-image markers. The detection logic below is unchanged.
    combined_z = np.zeros((8, 8))

    for ch in channels:
        means = get_grid_stats(ch, (8, 8), 'mean')
        Z = means.flatten()
        C, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
        plane = (A @ C).reshape(8, 8)
        diff = np.abs(means - plane)
        ch_z = robust_z_score(diff)
        combined_z = np.maximum(combined_z, ch_z)
        flagged_patches += int(np.sum(ch_z > 3.0))

    fraction = flagged_patches / (64 * 3)
    if fraction > 0.03:
        return {"name": "Lighting Consistency", "status": "inconclusive",
                "detail": "Deviations detected. However, real scenes with multiple light sources "
                          "also break a simple linear gradient, so this signal is weak on its own."}, combined_z
    return {"name": "Lighting Consistency", "status": "ok",
            "detail": "Lighting gradients follow natural linear constraints."}, combined_z


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

        # The primary model decides. The secondary is reported for transparency but
        # does NOT change the status.
        #
        # Measured over the sample images in this repo's testing, the primary was
        # right 5/5 while the secondary was right 3/5 -- and wrong in both
        # directions: it missed an obvious AI generation (0.02 artificial) and
        # called a real family photo artificial at 0.95. Letting a detector with
        # that record override the primary turned two correct answers into
        # "unclear", which is worse than not consulting it at all.
        disagree = (score_a >= 0.5) != (score_b >= 0.5)
        second = (f" Second opinion: {score_b:.2f}"
                  + (" (disagrees; it is the less reliable of the two)." if disagree else " (agrees)."))

        if score_a >= 0.65:
            return {"name": "AI Classifier", "status": "suspicious",
                    "detail": f"Detector flags artificial generation patterns ({score_a:.2f})."
                              + second + domain_note}
        if score_a <= 0.35:
            return {"name": "AI Classifier", "status": "ok",
                    "detail": f"Consistent with a human-made image (artificial-score {score_a:.2f})."
                              + second + domain_note}
        return {"name": "AI Classifier", "status": "inconclusive",
                "detail": f"The detector is undecided ({score_a:.2f}). Reported as unclear "
                          f"rather than guessing." + second + domain_note}
    except Exception as e:
        return {"name": "AI Classifier", "status": "inconclusive", "detail": f"Failed to classify image: {str(e)}"}


# ============================================================
# SUSPICIOUS REGIONS
# ------------------------------------------------------------
# This replaces the old colour heatmap. It uses exactly the same z-score grids
# the forensic checks already produce; instead of painting them over the image,
# it turns the flagged cells into a handful of labelled boxes, so the UI can put
# a numbered marker on each one with a one-sentence explanation.
# ============================================================

REGION_REASONS = {
    "ela": "Possible editing detected in this area.",
    "noise": "This area has a different noise pattern from the rest of the image.",
    "lighting": "Lighting in this area does not fully match the surrounding image.",
    "ela+noise": "Unusual pixel pattern detected here.",
}

# Same measurement, wording matched to the weight of the evidence. When every
# other check says the picture is an ordinary photo, these boxes are almost always
# a blurred background, a sharp in-focus subject or a flat wall -- all of which
# genuinely respond differently to the checks. Saying "possible editing" there
# contradicts the verdict printed directly above it.
REGION_REASONS_MILD = {
    "ela": "This area has much sharper detail than the rest, which is normal around a "
           "subject in focus.",
    "noise": "This area has a slightly different noise pattern, which is common in ordinary photos.",
    "lighting": "Lighting here differs a little from the area around it.",
    "ela+noise": "This area looks a little different from the rest of the picture.",
}


def soften_regions(regions, level):
    """Re-word regions when the overall verdict is that the picture looks real."""
    if level != "success":
        return regions
    back = {v: k for k, v in REGION_REASONS.items()}
    for r in regions:
        key = back.get(r["reason"])
        if key:
            r["reason"] = REGION_REASONS_MILD[key]
    return regions


def _connected_cells(mask):
    """Group touching flagged grid cells into clusters, so one edited patch
    becomes one marker instead of five separate ones."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    groups = []
    for i in range(h):
        for j in range(w):
            if not mask[i, j] or seen[i, j]:
                continue
            stack, cells = [(i, j)], []
            seen[i, j] = True
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            groups.append(cells)
    return groups


def _grid_to_boxes(z_grid, source, W, H, threshold=3.0):
    if z_grid is None:
        return []
    mask = z_grid > threshold
    if not mask.any():
        return []
    # Same guard the heatmap used: if nearly every cell is flagged, the cause is
    # global (whole-image re-compression or a filter), not a local edit to point at.
    if mask.sum() / mask.size > 0.25:
        return []

    gh, gw = z_grid.shape
    ph, pw = H / gh, W / gw
    boxes = []
    for cells in _connected_cells(mask):
        rows = [c[0] for c in cells]
        cols = [c[1] for c in cells]
        boxes.append({
            "x": int(min(cols) * pw),
            "y": int(min(rows) * ph),
            "width": int((max(cols) - min(cols) + 1) * pw),
            "height": int((max(rows) - min(rows) + 1) * ph),
            "sources": {source},
            "score": float(max(z_grid[r, c] for r, c in cells)),
        })
    return boxes


def _overlap_ratio(a, b):
    ix = max(0, min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"]))
    iy = max(0, min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"]))
    inter = ix * iy
    if inter == 0:
        return 0.0
    area_a = a["width"] * a["height"]
    area_b = b["width"] * b["height"]
    union = area_a + area_b - inter
    iou = inter / union if union else 0.0
    # A small box sitting inside a big one is the same finding, even though its
    # IoU is low, so treat containment as an overlap too.
    containment = inter / min(area_a, area_b) if min(area_a, area_b) else 0.0
    return max(iou, containment)


# The lighting check is weak on its own -- its own result text says real scenes
# with several light sources break the model. At the shared 3.0 threshold it
# produces markers on ordinary photos, so it needs stronger evidence to place one
# and is capped, to keep markers meaningful rather than decorative.
LIGHTING_THRESHOLD = 4.5
MAX_LIGHTING_ONLY_REGIONS = 2


def extract_suspicious_regions(ela_z, noise_z, light_z, W, H, max_regions=6,
                               ela_threshold=3.0):
    """Turn the forensic z-score grids into at most `max_regions` labelled boxes.

    `ela_threshold` is raised for video frames, where re-compression response is
    naturally noisier than on a still photo (see video_forensics).
    """
    strong_boxes = (_grid_to_boxes(ela_z, "ela", W, H, threshold=ela_threshold)
                    + _grid_to_boxes(noise_z, "noise", W, H))

    # Lighting only earns a marker when a stronger check already found something.
    # On its own it fires on ordinary photos, which would mean putting "suspicious"
    # markers on an image the verdict calls authentic.
    light_boxes = []
    if strong_boxes:
        light_boxes = _grid_to_boxes(light_z, "lighting", W, H, threshold=LIGHTING_THRESHOLD)
        light_boxes.sort(key=lambda d: -d["score"])
        light_boxes = light_boxes[:MAX_LIGHTING_ONLY_REGIONS]

    boxes = strong_boxes + light_boxes

    # Merge boxes that overlap, so a spot flagged by two different checks becomes
    # one marker whose explanation reflects both.
    merged = []
    for box in sorted(boxes, key=lambda d: -d["score"]):
        for m in merged:
            if _overlap_ratio(box, m) > 0.3:
                x0, y0 = min(m["x"], box["x"]), min(m["y"], box["y"])
                x1 = max(m["x"] + m["width"], box["x"] + box["width"])
                y1 = max(m["y"] + m["height"], box["y"] + box["height"])
                m.update(x=x0, y=y0, width=x1 - x0, height=y1 - y0)
                m["sources"] |= box["sources"]
                m["score"] = max(m["score"], box["score"])
                break
        else:
            merged.append(box)

    merged.sort(key=lambda d: -d["score"])
    regions = []
    for n, m in enumerate(merged[:max_regions], start=1):
        src = m["sources"]
        if "ela" in src and "noise" in src:
            key = "ela+noise"
        elif "ela" in src:
            key = "ela"
        elif "noise" in src:
            key = "noise"
        else:
            key = "lighting"
        regions.append({
            "id": n,
            "x": m["x"], "y": m["y"], "width": m["width"], "height": m["height"],
            "reason": REGION_REASONS[key],
            "strength": round(m["score"], 1),
        })
    return regions


# ============================================================
# EXPLANATION LAYER (Gemini)
# ============================================================

def generate_explanation(checks, file_path):
    try:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        # Treat the untouched .env placeholder as "no key set", so an unedited
        # .env falls back cleanly instead of firing a doomed API call.
        if api_key == "PASTE_YOUR_KEY_HERE":
            api_key = ""
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
    except Exception as e:
        # Print why we fell back. Without this the template summary looks identical
        # whether the key is missing, rejected, or the network blocked the call.
        print(f"[Gemini unavailable -> using template summary] {type(e).__name__}: {str(e)[:300]}")
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

# ============================================================
# PLAIN ENGLISH LAYER
# ------------------------------------------------------------
# The checks above keep their technical wording (it is still returned under
# "technical"). Everything here only rewrites those results into sentences a
# non-expert can read, without changing any measurement or verdict.
# ============================================================

CHECK_LABELS = {
    "Metadata Check": ("metadata", "Metadata"),
    "C2PA Check": ("c2pa", "Authenticity Record"),
    "Error Level Analysis": ("pixel_analysis", "Pixel Analysis"),
    "Noise Residual": ("noise", "Noise Pattern"),
    "Lighting Consistency": ("lighting", "Lighting"),
    "AI Classifier": ("ai_detection", "AI Detection"),
}

# status -> (simple status, short label shown on the card, icon)
STATUS_SIMPLE = {
    "ok":           ("normal", "Normal", "check"),
    "suspicious":   ("suspicious", "Suspicious", "warn"),
    "inconclusive": ("unclear", "Unclear", "question"),
    "unavailable":  ("none", "Not available", "dash"),
    "removed":      ("none", "Not found", "dash"),
}

PLAIN_MESSAGES = {
    ("metadata", "ok"): "Normal camera information was found.",
    ("metadata", "removed"): "No camera information was found, which is normal for images shared online.",
    ("metadata", "suspicious"): "The image information names editing or AI software.",
    ("metadata", "inconclusive"): "Some information was found, but it does not name a camera or an app.",

    ("c2pa", "ok"): "An authenticity record was found, with no sign of AI.",
    ("c2pa", "suspicious"): "The authenticity record says this was made with AI.",
    ("c2pa", "unavailable"): "No authenticity information was found. Most images do not have any.",

    ("pixel_analysis", "ok"): "No unusual areas were found.",
    ("pixel_analysis", "suspicious"): "Some unusual areas were detected.",
    ("pixel_analysis", "inconclusive"): "This check could not give a clear answer for this image.",

    ("noise", "ok"): "Looks consistent across the image.",
    ("noise", "suspicious"): "Some areas do not match the rest of the image.",
    ("noise", "inconclusive"): "The image looks smoothed, so this check could not compare areas.",

    ("lighting", "ok"): "Looks natural.",
    ("lighting", "inconclusive"): "Small differences were found, but real photos often show these too.",
    ("lighting", "suspicious"): "The lighting does not look consistent.",

    # Do not claim the two detectors agree: the primary decides on its own, and
    # the secondary is often the one that is wrong.
    ("ai_detection", "ok"): "The detector sees a human-made picture.",
    ("ai_detection", "suspicious"): "The detector found signs of AI generation.",
    ("ai_detection", "inconclusive"): "The detector could not decide either way.",
}

# The simple one-line verdicts. Left side is what build_verdict() already returns,
# so the decision logic stays in one place and is not duplicated here.
VERDICT_SIMPLE = {
    "LIKELY AI GENERATED / MANIPULATED": ("LIKELY AI",
        "The AI detector found strong signs this was generated, and other checks agree."),
    "POSSIBLY AI GENERATED": ("LIKELY AI",
        "The AI detector found signs this picture was generated."),
    "LIKELY MANIPULATED / EDITED": ("MAYBE EDITED",
        "Parts of this image look like they may have been changed."),
    "WEAK ANOMALY DETECTED": ("NOT SURE",
        "One area looked different from the rest, which is often harmless."),
    "LIKELY AUTHENTIC": ("LOOKS REAL",
        "This shows the normal patterns of an ordinary photo."),
    "INCONCLUSIVE / UNCERTAIN": ("NOT SURE",
        "There is not enough evidence to give a clear answer."),
}


def _blocked_checks(checks):
    """Which checks could not run at all, and why - in plain words.

    This matters for the verdict wording: a screenshot or a PNG has no EXIF, no
    C2PA record and no JPEG history, so three of the six checks are silent for
    reasons that have nothing to do with whether the picture is genuine. Saying
    'not sure' without saying why reads as suspicion the evidence does not support.
    """
    by = {c["name"]: c for c in checks}
    reasons = []
    if by.get("Metadata Check", {}).get("status") == "removed":
        reasons.append("it carries no camera information")
    ela = by.get("Error Level Analysis", {})
    if ela.get("status") == "inconclusive" and "not a JPEG" in ela.get("detail", ""):
        reasons.append("the pixel check only works on JPEG photos")
    if by.get("C2PA Check", {}).get("status") == "unavailable":
        reasons.append("it has no authenticity record")
    return reasons


def simplify_verdict(verdict_status, verdict_level, checks=None):
    result, summary = VERDICT_SIMPLE.get(verdict_status, ("NOT SURE",
        "There is not enough evidence to give a clear answer."))
    checks = checks or []

    ai = next((c for c in checks if c["name"] == "AI Classifier"), None)
    ai_says_human = bool(ai and ai["status"] == "ok")

    # A single weak flag, with the AI detectors agreeing the picture looks
    # human-made, is much closer to "fine" than to "unknown". Calling both of
    # those "NOT SURE" was treating ordinary photos as suspicious.
    if verdict_status == "WEAK ANOMALY DETECTED" and ai_says_human:
        result, verdict_level = "LOOKS REAL", "success"
        summary = ("The AI detector sees a human-made picture and most checks look normal. "
                   "One area looked a little different, which is usually harmless.")

    note = ""
    blocked = _blocked_checks(checks)
    if blocked and result in ("NOT SURE", "LOOKS REAL"):
        if len(blocked) >= 2:
            note = ("Some checks could not run on this file because "
                    + ", ".join(blocked[:-1]) + " and " + blocked[-1]
                    + ". That is normal for screenshots and images saved from the web, "
                      "and it is not a sign of editing.")
        else:
            note = ("One check could not run because " + blocked[0]
                    + ". That is normal for screenshots and images saved from the web.")

    return result, summary, verdict_level, note


def simplify_check(check):
    key, label = CHECK_LABELS.get(check["name"], (check["name"].lower(), check["name"]))
    simple, status_label, icon = STATUS_SIMPLE.get(check["status"], ("unclear", "Unclear", "question"))
    message = PLAIN_MESSAGES.get((key, check["status"]))
    if message is None:
        # Unmapped combination: say plainly that it was unclear rather than
        # falling back to the technical sentence the user should not have to read.
        message = "This check did not give a clear answer."
    # ELA skips non-JPEG images entirely; saying so is clearer than "unclear".
    if key == "pixel_analysis" and "not a JPEG" in check.get("detail", ""):
        message = "This check only works on JPEG photos, so it was skipped."
    return {
        "key": key,
        "label": label,
        "status": simple,
        "status_label": status_label,
        "icon": icon,
        "message": message,
        "technical": f"{check['name']}: {check['status'].upper()} - {check['detail']}",
    }


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def analyze_image(file_path, progress=None):
    """Run the full forensic pipeline and return a result shaped for the UI.

    `progress` is an optional callable(percent, message) used to drive the
    scanning screen with real progress rather than a fake timer.
    """
    def step(pct, msg):
        if progress:
            progress(pct, msg)

    step(4, "Opening media...")
    img_pil = Image.open(file_path)
    img_format = img_pil.format
    width, height = img_pil.size

    checks = []

    step(12, "Checking metadata...")
    checks.append(check_metadata(img_pil))

    step(22, "Checking authenticity record...")
    checks.append(check_c2pa(file_path))

    step(36, "Analyzing pixel patterns...")
    ela_res, ela_z = check_ela(file_path, img_format)
    checks.append(ela_res)

    step(50, "Checking noise consistency...")
    noise_res, noise_z = check_noise_residual(file_path)
    checks.append(noise_res)

    step(62, "Checking lighting...")
    light_res, light_z = check_lighting(file_path)
    checks.append(light_res)

    step(74, "Running AI detection...")
    checks.append(check_ai_classifier(img_pil))

    step(86, "Identifying suspicious regions...")
    verdict_status, verdict_level, verdict_detail = build_verdict(checks)
    regions = extract_suspicious_regions(ela_z, noise_z, light_z, width, height)

    step(93, "Preparing results...")
    report_text, report_source = generate_explanation(checks, file_path)

    overall, summary, level, note = simplify_verdict(verdict_status, verdict_level, checks)
    regions = soften_regions(regions, level)

    step(100, "Done")
    return {
        "media_type": "image",
        "overall_result": overall,
        "overall_level": level,
        "summary": summary,
        "note": note,
        "checks": [simplify_check(c) for c in checks],
        "suspicious_points": regions,
        "image": {"width": width, "height": height, "format": img_format},
        "technical": {
            "verdict_raw": verdict_status,
            "verdict_detail": verdict_detail,
            "checks_raw": checks,
            "explanation": report_text,
            "explanation_source": report_source,
        },
    }
