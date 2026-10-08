#!/usr/bin/env python3
"""
Auto Music Channel
------------------
Generates one AI music video (music + background image + thumbnail + title/description)
and uploads it to YouTube. Run by GitHub Actions twice a day.

All secrets come from environment variables (GitHub Secrets). Nothing is hard-coded.
"""
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.io import wavfile

# ----------------------------- SETTINGS --------------------------------------
OUT = Path(os.getenv("WORK_DIR", "output"))
DRY_RUN = os.getenv("DRY_RUN", "false").strip().lower() == "true"
FAKE_MUSIC = os.getenv("FAKE_MUSIC", "false").strip().lower() == "true"  # quick pipeline test
MIN_MINUTES = int(os.getenv("MIN_MINUTES", "5"))
MAX_MINUTES = int(os.getenv("MAX_MINUTES", "40"))
MAX_CLIPS = int(os.getenv("MAX_CLIPS", "8"))       # unique AI clips per video (more = slower)
CLIP_SECONDS = 30
CROSSFADE_SECONDS = 3
SAMPLE_RATE = 32000                                 # MusicGen output rate
PRIVACY = os.getenv("UPLOAD_PRIVACY", "public")     # public / unlisted / private
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Niches are all in the same "chill / study / relax" family, so the channel stays consistent.
NICHES = [
    {"name": "Lofi Hip Hop",
     "music": "lofi hip hop beat, mellow electric piano, vinyl crackle, soft drums, warm bass, relaxed",
     "scene": "cozy bedroom desk at night, warm lamp light, rain on the window, plants, illustration style"},
    {"name": "Chillhop",
     "music": "chillhop instrumental, jazzy guitar chords, laid-back drums, smooth bass, calm",
     "scene": "quiet city street at dusk with soft neon lights, cafe windows glowing, illustration style"},
    {"name": "Jazz Lofi",
     "music": "lofi jazz, soft saxophone, brushed drums, upright bass, late night cafe mood",
     "scene": "empty cozy jazz cafe at night, warm lights, coffee cup on table, illustration style"},
    {"name": "Rainy Day Lofi",
     "music": "rainy day lofi, gentle piano, soft rain ambience, slow tempo, melancholic and calm",
     "scene": "window with heavy rain drops, blurry city lights outside, cozy blanket, illustration style"},
    {"name": "Ambient Study Music",
     "music": "ambient study music, soft synth pads, slow evolving texture, peaceful, no drums",
     "scene": "calm mountain lake at sunrise, soft mist, pastel colors, digital painting"},
    {"name": "Sleep Piano",
     "music": "slow soft piano, dreamy reverb, gentle ambient pads, very relaxing, sleep music",
     "scene": "starry night sky over quiet hills, crescent moon, deep blue and purple, digital painting"},
    {"name": "Synthwave Chill",
     "music": "chill synthwave, warm analog synths, slow retro beat, dreamy, night drive mood",
     "scene": "retro sunset road with palm trees, purple and orange sky, synthwave illustration"},
    {"name": "Bossa Nova Lofi",
     "music": "bossa nova lofi, nylon guitar, soft percussion, warm and sunny, relaxed",
     "scene": "sunny balcony with plants and ocean view, warm afternoon light, illustration style"},
]
EXTRA_FLAVORS = ["with soft vinyl crackle", "with gentle bell tones", "with warm rhodes keys",
                 "with a mellow guitar melody", "with a slow dreamy melody", "with light tape hiss",
                 "with soft strings in the background", "with a smooth bass groove"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------- METADATA --------------------------------------
def fallback_metadata(niche, minutes):
    mood = random.choice(["Calm", "Cozy", "Dreamy", "Peaceful", "Late Night", "Soft"])
    use = random.choice(["Study", "Relax", "Focus", "Sleep", "Work"])
    return {
        "title": f"{mood} {niche['name']} to {use} To | {minutes} Minute Mix",
        "thumbnail_text": f"{mood} {niche['name']}",
        "description": (f"{niche['name']} music for studying, relaxing, working, or sleeping.\n\n"
                        "Music is AI-generated."),
        "tags": [niche["name"].lower(), "lofi", "study music", "relaxing music", "chill beats",
                 "focus music", "background music", "ai music"],
        "image_prompt": niche["scene"],
    }


def gemini_metadata(niche, minutes):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    prompt = f"""You write YouTube metadata for a relaxing instrumental music channel.
This video: genre "{niche['name']}", length about {minutes} minutes, no vocals.
Return ONLY JSON with these keys:
"title": catchy, max 80 characters, English, no emojis, no clickbait lies,
"thumbnail_text": 2 to 4 words, short and bold,
"description": 3 short paragraphs (what it is, best moments to listen, a friendly call to subscribe), plain text, and a final line "Music is AI-generated.",
"tags": array of 12 relevant tags,
"image_prompt": one sentence describing a beautiful calm background picture that matches the genre. Scene: {niche['scene']}. No text, no people's faces, no logos."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    r = requests.post(
        url, headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"responseMimeType": "application/json", "temperature": 1.0}},
        timeout=90)
    r.raise_for_status()
    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    data = json.loads(text)
    for k in ("title", "thumbnail_text", "description", "tags", "image_prompt"):
        if k not in data:
            raise ValueError(f"missing {k}")
    return data


def get_metadata(niche, minutes):
    try:
        data = gemini_metadata(niche, minutes)
        if data:
            log("Metadata from Gemini OK")
        else:
            log("No GEMINI_API_KEY, using template metadata")
            data = fallback_metadata(niche, minutes)
    except Exception as e:
        log(f"Gemini failed ({type(e).__name__}), using template metadata")
        data = fallback_metadata(niche, minutes)
    data["title"] = re.sub(r"[<>]", "", str(data["title"]))[:95].strip()
    data["description"] = str(data["description"])[:4500]
    if "AI-generated" not in data["description"]:
        data["description"] += "\n\nMusic is AI-generated."
    tags, total = [], 0
    for t in data["tags"]:
        t = re.sub(r"[<>,]", "", str(t)).strip()
        if t and total + len(t) + 1 < 450:
            tags.append(t)
            total += len(t) + 1
    data["tags"] = tags
    return data


# ----------------------------- IMAGES ----------------------------------------
def gradient_background(path):
    log("Making a gradient background (fallback)")
    h, w = 1080, 1920
    c1 = np.array([random.randint(20, 120) for _ in range(3)], dtype=np.float32)
    c2 = np.array([random.randint(60, 220) for _ in range(3)], dtype=np.float32)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img = (c1 * (1 - t) + c2 * t) * np.ones((h, w, 3), dtype=np.float32)
    Image.fromarray(img.astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)).save(path, quality=92)


def make_background(prompt, path):
    full = f"{prompt}, highly detailed, soft lighting, wallpaper, no text"
    for attempt in range(3):
        try:
            seed = random.randint(1, 10**6)
            url = "https://image.pollinations.ai/prompt/" + quote(full)
            r = requests.get(url, params={"width": 1920, "height": 1080, "nologo": "true",
                                          "seed": seed}, timeout=180)
            r.raise_for_status()
            if not r.headers.get("content-type", "").startswith("image"):
                raise ValueError("not an image")
            tmp = path.with_suffix(".raw")
            tmp.write_bytes(r.content)
            img = Image.open(tmp).convert("RGB")
            tmp.unlink()
            if img.width < 800:
                raise ValueError("image too small")
            ratio = max(1920 / img.width, 1080 / img.height)
            img = img.resize((math.ceil(img.width * ratio), math.ceil(img.height * ratio)), Image.LANCZOS)
            left, top = (img.width - 1920) // 2, (img.height - 1080) // 2
            img.crop((left, top, left + 1920, top + 1080)).save(path, quality=92)
            log("Background image OK")
            return
        except Exception as e:
            log(f"Image attempt {attempt + 1} failed: {type(e).__name__}")
            time.sleep(5)
    gradient_background(path)


def find_font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "C:/Windows/Fonts/arialbd.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def make_thumbnail(bg_path, text, path):
    img = Image.open(bg_path).convert("RGB").resize((1280, 720), Image.LANCZOS)
    # dark gradient on the left so text is readable
    overlay = np.zeros((720, 1280, 4), dtype=np.uint8)
    alpha = np.clip(200 - np.linspace(0, 260, 1280), 0, 200).astype(np.uint8)
    overlay[:, :, 3] = alpha[None, :]
    img = Image.alpha_composite(img.convert("RGBA"), Image.fromarray(overlay, "RGBA")).convert("RGB")
    draw = ImageDraw.Draw(img)
    words = text.upper().split()
    size = 120
    font = find_font(size)
    lines = []
    while size >= 50:
        font = find_font(size)
        lines, cur = [], ""
        for w in words:
            test = (cur + " " + w).strip()
            if draw.textlength(test, font=font) <= 760:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        if len(lines) * size * 1.15 <= 560:
            break
        size -= 10
    total_h = len(lines) * int(size * 1.15)
    y = (720 - total_h) // 2
    for line in lines:
        draw.text((64, y + 4), line, font=font, fill=(0, 0, 0))
        draw.text((60, y), line, font=font, fill=(255, 255, 255))
        y += int(size * 1.15)
    img.save(path, "JPEG", quality=88)
    log("Thumbnail OK")


# ----------------------------- MUSIC -----------------------------------------
_model = {}


def fake_clip():
    t = np.linspace(0, CLIP_SECONDS, CLIP_SECONDS * SAMPLE_RATE, endpoint=False, dtype=np.float32)
    f = random.choice([220, 261.6, 329.6, 392])
    return 0.2 * np.sin(2 * np.pi * f * t) + 0.1 * np.sin(2 * np.pi * f * 1.5 * t)


def generate_clip(prompt):
    if FAKE_MUSIC:
        return fake_clip()
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    if not _model:
        log("Loading MusicGen model (first time downloads ~2 GB)...")
        torch.set_num_threads(os.cpu_count() or 2)
        _model["p"] = AutoProcessor.from_pretrained("facebook/musicgen-small")
        _model["m"] = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        _model["m"].eval()
    inputs = _model["p"](text=[prompt], padding=True, return_tensors="pt")
    with torch.no_grad():
        audio = _model["m"].generate(**inputs, do_sample=True, guidance_scale=3.0,
                                     max_new_tokens=CLIP_SECONDS * 50)
    return audio[0, 0].cpu().numpy().astype(np.float32)


def level(clip):
    rms = float(np.sqrt(np.mean(clip ** 2))) + 1e-9
    return np.clip(clip * (0.12 / rms), -1.0, 1.0)


def build_track(clips, target_samples):
    xf = CROSSFADE_SECONDS * SAMPLE_RATE
    seq, total = [], 0
    while total < target_samples:
        order = list(range(len(clips)))
        random.shuffle(order)
        if seq and len(order) > 1 and order[0] == seq[-1]:
            order.append(order.pop(0))
        for i in order:
            seq.append(i)
            total += len(clips[i]) - xf
            if total >= target_samples:
                break
    fade_in = np.linspace(0, 1, xf, dtype=np.float32)
    pieces, prev_tail = [], None
    for i in seq:
        c = clips[i]
        head, body, tail = c[:xf], c[xf:-xf], c[-xf:]
        pieces.append(head if prev_tail is None else prev_tail * (1 - fade_in) + head * fade_in)
        pieces.append(body)
        prev_tail = tail
    pieces.append(prev_tail)
    out = np.concatenate(pieces)[:target_samples]
    fi, fo = 3 * SAMPLE_RATE, 6 * SAMPLE_RATE
    out[:fi] *= np.linspace(0, 1, fi, dtype=np.float32)
    out[-fo:] *= np.linspace(1, 0, fo, dtype=np.float32)
    peak = float(np.max(np.abs(out))) + 1e-9
    return out * (0.9 / peak)


def make_music(niche, seconds, path):
    n = max(1, min(MAX_CLIPS, math.ceil(seconds / CLIP_SECONDS)))
    log(f"Generating {n} unique clips, then arranging them into {seconds // 60} min {seconds % 60} s")
    clips = []
    for i in range(n):
        prompt = f"{niche['music']}, {random.choice(EXTRA_FLAVORS)}, instrumental, no vocals"
        t0 = time.time()
        clips.append(level(generate_clip(prompt)))
        log(f"  clip {i + 1}/{n} done in {int(time.time() - t0)}s")
    track = build_track(clips, seconds * SAMPLE_RATE)
    wavfile.write(path, SAMPLE_RATE, (track * 32767).astype(np.int16))
    log("Music track OK")


# ----------------------------- VIDEO -----------------------------------------
def make_video(img_path, audio_path, video_path):
    log("Rendering video with FFmpeg...")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", "5", "-i", str(img_path),
           "-i", str(audio_path), "-c:v", "libx264", "-tune", "stillimage", "-preset", "veryfast",
           "-crf", "30", "-pix_fmt", "yuv420p", "-vf", "scale=1920:1080", "-r", "5",
           "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2", "-shortest",
           "-movflags", "+faststart", str(video_path)]
    subprocess.run(cmd, check=True)
    log(f"Video OK ({video_path.stat().st_size / 1e6:.0f} MB)")


# ----------------------------- UPLOAD ----------------------------------------
def upload(video_path, thumb_path, meta):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    creds = Credentials(None, refresh_token=os.environ["YT_REFRESH_TOKEN"],
                        token_uri="https://oauth2.googleapis.com/token",
                        client_id=os.environ["YT_CLIENT_ID"],
                        client_secret=os.environ["YT_CLIENT_SECRET"],
                        scopes=["https://www.googleapis.com/auth/youtube.upload"])
    creds.refresh(Request())
    yt = build("youtube", "v3", credentials=creds, cache_discovery=False)
    body = {
        "snippet": {"title": meta["title"], "description": meta["description"],
                    "tags": meta["tags"], "categoryId": "10"},
        "status": {"privacyStatus": PRIVACY, "selfDeclaredMadeForKids": False,
                   "containsSyntheticMedia": True},
    }
    media = MediaFileUpload(str(video_path), chunksize=8 * 1024 * 1024, resumable=True)
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = None
    while resp is None:
        status, resp = req.next_chunk()
        if status:
            log(f"  uploaded {int(status.progress() * 100)}%")
    vid = resp["id"]
    log(f"Uploaded: https://youtu.be/{vid}")
    try:
        yt.thumbnails().set(videoId=vid, media_body=MediaFileUpload(str(thumb_path))).execute()
        log("Thumbnail set OK")
    except Exception as e:
        log(f"Thumbnail failed (is your channel verified?): {type(e).__name__}")
    return vid


# ----------------------------- MAIN ------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    niche = random.choice(NICHES)
    minutes = random.randint(MIN_MINUTES, MAX_MINUTES)
    seconds = minutes * 60
    log(f"Niche: {niche['name']} | Length: {minutes} min | Dry run: {DRY_RUN}")

    meta = get_metadata(niche, minutes)
    log(f"Title: {meta['title']}")
    bg, thumb = OUT / "background.jpg", OUT / "thumbnail.jpg"
    audio, video = OUT / "music.wav", OUT / "video.mp4"

    make_background(meta["image_prompt"], bg)
    make_thumbnail(bg, meta["thumbnail_text"], thumb)
    make_music(niche, seconds, audio)
    make_video(bg, audio, video)
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if DRY_RUN:
        log("Dry run: skipping upload. Check the files in the output folder.")
        return
    upload(video, thumb, meta)
    for f in (audio, video):  # free disk space
        f.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FAILED: {type(e).__name__}: {e}")
        sys.exit(1)
