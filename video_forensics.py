"""Video analysis for PixelSleuth.

This module does not re-implement any forensics. It samples frames out of a
video and hands each frame to the same checks that already run on still images
(forensics.check_noise_residual, check_lighting, check_ai_classifier), then
aggregates the per-frame answers into one result.

A note on pixel analysis (Error Level Analysis): it re-compresses the frame and
compares how different areas respond. On a still JPEG that doubles as evidence of
editing history, because the file carries its own compression history. A frame
decoded from an H.264/VP9 stream does not, so here the same measurement reads as
a texture/detail difference between areas rather than proof of a past edit. A
pasted region still stands out, but the signal is weaker than on a photo and the
results say so rather than overstating it.
"""

import base64
import os
import tempfile

import cv2
from PIL import Image

from forensics import (
    build_verdict,
    check_ai_classifier,
    check_c2pa,
    check_ela,
    check_lighting,
    check_noise_residual,
    extract_suspicious_regions,
    simplify_check,
)

# Each sampled frame runs two neural classifiers on the CPU, so this is the main
# driver of how long an analysis takes. 12 frames keeps a typical clip under a
# minute while still covering the whole timeline.
DEFAULT_MAX_FRAMES = 12
THUMB_MAX_WIDTH = 640


def _thumbnail(frame_bgr, max_width=THUMB_MAX_WIDTH):
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not ok:
        return None
    return base64.b64encode(buf).decode('utf-8')


def _format_timestamp(seconds):
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:01d}:{seconds % 60:02d}"


# A decoded frame's response to re-compression varies more across the picture
# than a still photo's does, because the video codec has already left block
# artefacts of its own. Scored at the stills cutoff of 3.0 this flags ordinary
# frames, so the same grid is judged at a stricter cutoff for video. The
# measurement is unchanged -- only the line between "normal" and "unusual" moves.
VIDEO_ELA_THRESHOLD = 5.0


def _restate_ela_for_video(ela_z):
    """Re-apply check_ela's own decision rule to its grid at the video cutoff."""
    if ela_z is None:
        return {"name": "Error Level Analysis", "status": "inconclusive",
                "detail": "Pixel analysis could not run on this frame."}
    extreme = int((ela_z > VIDEO_ELA_THRESHOLD).sum())
    fraction = extreme / ela_z.size
    if 0 < fraction < 0.25:
        return {"name": "Error Level Analysis", "status": "suspicious",
                "detail": f"Localized anomaly detected ({extreme} patches, video cutoff "
                          f"z>{VIDEO_ELA_THRESHOLD})."}
    if fraction >= 0.25:
        return {"name": "Error Level Analysis", "status": "inconclusive",
                "detail": "Differences are spread across the whole frame, which points at "
                          "video compression rather than a local edit."}
    return {"name": "Error Level Analysis", "status": "ok",
            "detail": f"No localized compression differences (video cutoff z>{VIDEO_ELA_THRESHOLD})."}


def _join_times(frames, limit=3):
    """'0:02, 0:05 and 0:09' - de-duplicated, since several sampled frames can
    round to the same second on a short clip."""
    seen = []
    for f in frames:
        if f["timestamp"] not in seen:
            seen.append(f["timestamp"])
    seen = seen[:limit]
    if len(seen) == 1:
        return seen[0]
    return ", ".join(seen[:-1]) + " and " + seen[-1]


def _frame_status(checks):
    """Collapse one frame's checks into normal / suspicious / unclear."""
    verdict_status, level, _ = build_verdict(checks)
    if level == "danger":
        return "suspicious", verdict_status
    if level == "warning":
        return "suspicious", verdict_status
    if level == "success":
        return "normal", verdict_status
    return "unclear", verdict_status


def _analyze_frame(frame_bgr, workdir, idx):
    """Run the image checks on a decoded video frame."""
    path = os.path.join(workdir, f"frame_{idx}.png")
    # PNG, not JPEG: writing a JPEG here would add compression artefacts of our
    # own on top of the video codec's, which the noise check would then measure.
    cv2.imwrite(path, frame_bgr)
    jpg_path = os.path.join(workdir, f"frame_{idx}.jpg")
    cv2.imwrite(jpg_path, frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    try:
        checks = []

        # Pixel analysis on a frame measures how differently each area responds
        # to re-compression. On a still JPEG that doubles as evidence of editing
        # history; a decoded frame has no such history, so this reads as a
        # texture/detail difference between areas. It is a real measurement and a
        # pasted region still stands out, but it is weaker evidence than on a
        # photo, which is why the technical note spells the difference out.
        _, ela_z = check_ela(jpg_path, 'JPEG')
        checks.append(_restate_ela_for_video(ela_z))

        noise_res, noise_z = check_noise_residual(path)
        checks.append(noise_res)
        light_res, light_z = check_lighting(path)
        checks.append(light_res)
        checks.append(check_ai_classifier(Image.open(path)))

        h, w = frame_bgr.shape[:2]
        regions = extract_suspicious_regions(ela_z, noise_z, light_z, w, h,
                                             ela_threshold=VIDEO_ELA_THRESHOLD)
        return checks, regions
    finally:
        for p in (path, jpg_path):
            if os.path.exists(p):
                os.remove(p)


def analyze_video(file_path, progress=None, max_frames=DEFAULT_MAX_FRAMES):
    def step(pct, msg):
        if progress:
            progress(pct, msg)

    step(3, "Opening video...")
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError("This video file could not be opened. The format may not be supported.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0 or fps > 240:
        fps = 25.0  # some containers report nonsense; assume a sane default
    duration = (total_frames / fps) if total_frames > 0 else 0.0

    step(6, "Extracting frames...")
    if total_frames > 0:
        count = min(max_frames, total_frames)
        indices = [int(i * (total_frames - 1) / max(1, count - 1)) for i in range(count)] \
            if count > 1 else [0]
    else:
        # Frame count unknown (streamed/variable containers): read sequentially.
        indices = list(range(max_frames))

    workdir = tempfile.mkdtemp(prefix="pixelsleuth_video_")
    frames = []
    try:
        for n, frame_index in enumerate(indices, start=1):
            pct = 8 + int(78 * (n - 1) / max(1, len(indices)))
            step(pct, f"Analyzing frame {n} of {len(indices)}...")

            if total_frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            checks, regions = _analyze_frame(frame, workdir, n)
            status, raw_verdict = _frame_status(checks)
            timestamp = frame_index / fps
            fh, fw = frame.shape[:2]

            frames.append({
                "index": n,
                "frame_number": int(frame_index),
                # Region boxes are in full-frame pixels while the thumbnail is
                # scaled down, so the UI needs the original size to place markers.
                "frame_width": int(fw),
                "frame_height": int(fh),
                "time_seconds": round(timestamp, 2),
                "timestamp": _format_timestamp(timestamp),
                "status": status,
                "checks": [simplify_check(c) for c in checks],
                "suspicious_points": regions,
                "thumbnail": _thumbnail(frame),
                "technical": {"verdict_raw": raw_verdict,
                              "checks_raw": checks},
            })
    finally:
        cap.release()
        try:
            os.rmdir(workdir)
        except OSError:
            pass

    if not frames:
        raise ValueError("No frames could be read from this video.")

    step(90, "Checking authenticity record...")
    c2pa_check = check_c2pa(file_path)

    step(95, "Preparing results...")
    result = _summarise(frames, c2pa_check, duration, fps, total_frames, len(indices))
    step(100, "Done")
    return result


def _summarise(frames, c2pa_check, duration, fps, total_frames, sampled):
    analyzed = len(frames)
    suspicious = [f for f in frames if f["status"] == "suspicious"]
    normal = [f for f in frames if f["status"] == "normal"]
    unclear = [f for f in frames if f["status"] == "unclear"]

    susp_ratio = len(suspicious) / analyzed

    # Did the AI detector itself flag frames, or is this only edit-type evidence?
    ai_flagged = sum(
        1 for f in frames
        for c in f["checks"]
        if c["key"] == "ai_detection" and c["status"] == "suspicious"
    )

    if ai_flagged / analyzed >= 0.34:
        overall, level = "MAYBE AI-MADE", "danger"
        summary = "Several analyzed frames show signs of AI generation."
    elif susp_ratio >= 0.34:
        overall, level = "MAYBE EDITED", "warning"
        summary = "Several analyzed frames contain unusual areas."
    elif suspicious:
        overall, level = "NOT SURE", "neutral"
        summary = ("Most analyzed frames look normal, but unusual patterns were found around "
                   f"{_join_times(suspicious)}.")
    elif len(normal) >= max(1, int(analyzed * 0.6)):
        overall, level = "LOOKS REAL", "success"
        summary = "No significant suspicious patterns were detected in the analyzed frames."
    else:
        overall, level = "NOT SURE", "neutral"
        summary = "The results are unclear because the analyzed frames did not agree."

    if suspicious and overall in ("MAYBE AI-MADE", "MAYBE EDITED"):
        summary += f" The clearest examples are around {_join_times(suspicious, limit=4)}."

    # Roll the per-frame checks up into the same card set the image view uses, so
    # the UI stays identical. A card is suspicious if any frame flagged it.
    card_order = ["pixel_analysis", "noise", "lighting", "ai_detection"]
    cards = []
    for key in card_order:
        per_frame = [c for f in frames for c in f["checks"] if c["key"] == key]
        if not per_frame:
            continue
        flagged = sum(1 for c in per_frame if c["status"] == "suspicious")
        if flagged:
            card = dict(per_frame[0])
            card["status"], card["status_label"], card["icon"] = "suspicious", "Suspicious", "warn"
            card["message"] = f"Unusual results in {flagged} of {len(per_frame)} analyzed frames."
        else:
            unclear_n = sum(1 for c in per_frame if c["status"] == "unclear")
            card = dict(per_frame[0])
            if unclear_n > len(per_frame) / 2:
                card["status"], card["status_label"], card["icon"] = "unclear", "Unclear", "question"
                card["message"] = "The analyzed frames did not give a clear answer."
            else:
                card["status"], card["status_label"], card["icon"] = "normal", "Normal", "check"
                card["message"] = f"Consistent across all {len(per_frame)} analyzed frames."
        card["technical"] = f"{key}: {flagged} of {len(per_frame)} sampled frames flagged."
        cards.append(card)

    cards.insert(0, simplify_check(c2pa_check))

    return {
        "media_type": "video",
        "overall_result": overall,
        "overall_level": level,
        "summary": summary,
        "note": "",
        "checks": cards,
        "frames": frames,
        "video": {
            "duration_seconds": round(duration, 2),
            "duration_label": _format_timestamp(duration),
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "analyzed_frames": analyzed,
            "sampled_requested": sampled,
        },
        "counts": {
            "normal": len(normal),
            "suspicious": len(suspicious),
            "unclear": len(unclear),
        },
        "technical": {
            "note": (f"{analyzed} frames were sampled evenly across the video "
                     f"({total_frames} frames total at {round(fps, 2)} fps). "
                     "Pixel analysis on video frames measures how each area responds to "
                     "re-compression. Unlike a still JPEG, a decoded frame carries no original "
                     "compression history, so treat this signal as weaker than on a photo."),
        },
    }
