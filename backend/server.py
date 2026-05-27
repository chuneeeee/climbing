import os, uuid, shutil, threading, subprocess, traceback
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

JOBS_DIR = Path("/tmp/boulder_jobs")
JOBS_DIR.mkdir(exist_ok=True)

jobs: dict = {}
jobs_lock = threading.Lock()
job_semaphore = threading.Semaphore(2)

RATIO_MAP = {"9:16": 9/16, "1:1": 1.0, "4:5": 4/5}


def get_crop_box(landmarks, vid_w, vid_h, ratio, padding):
    pts = [(l.x, l.y) for l in landmarks if l.visibility > 0.3]
    if not pts:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    pw = (max_x - min_x) * vid_w
    ph = (max_y - min_y) * vid_h

    # Padded person bounds — these must NEVER be cut off
    px1 = max(0.0, min_x * vid_w - pw * padding)
    px2 = min(float(vid_w), max_x * vid_w + pw * padding)
    py1 = max(0.0, min_y * vid_h - ph * padding)
    py2 = min(float(vid_h), max_y * vid_h + ph * padding)

    padded_w = px2 - px1
    padded_h = py2 - py1
    cx = (px1 + px2) / 2
    cy = (py1 + py2) / 2

    # Crop = 3x padded person height (person in center 1/3)
    crop_h = padded_h * 3
    crop_w = crop_h * ratio

    # Ensure crop is wide enough to contain the person
    if crop_w < padded_w:
        crop_w = padded_w
        crop_h = crop_w / ratio

    # Clamp to video bounds while preserving ratio
    if crop_w > vid_w:
        crop_w = float(vid_w)
        crop_h = crop_w / ratio
    if crop_h > vid_h:
        crop_h = float(vid_h)
        crop_w = crop_h * ratio

    sx = max(0.0, min(cx - crop_w / 2, vid_w - crop_w))
    sy = max(0.0, min(cy - crop_h / 2, vid_h - crop_h))

    return {
        "sx": sx, "sy": sy, "cw": crop_w, "ch": crop_h,
        "px1": px1, "px2": px2, "py1": py1, "py2": py2,
    }


def constrain_to_person(sx, sy, cw, ch, px1, py1, px2, py2, vid_w, vid_h):
    """Shift crop box just enough so person bounding box is fully inside."""
    if px1 < sx:
        sx = px1
    if px2 > sx + cw:
        sx = px2 - cw
    if py1 < sy:
        sy = py1
    if py2 > sy + ch:
        sy = py2 - ch
    sx = max(0.0, min(sx, vid_w - cw))
    sy = max(0.0, min(sy, vid_h - ch))
    return sx, sy


def lerp_box(a, b, t):
    return {k: a[k] + t * (b[k] - a[k]) for k in a}


def get_box_at(t, key_times, keys):
    if t <= key_times[0]:
        return keys[0]
    if t >= key_times[-1]:
        return keys[-1]
    lo, hi = 0, len(key_times) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if key_times[mid] <= t:
            lo = mid
        else:
            hi = mid
    f = (t - key_times[lo]) / (key_times[hi] - key_times[lo])
    return lerp_box(keys[lo], keys[hi], f)


def process_job(job_id: str, input_path: str, ratio_str: str, padding: float, ema_smooth: int, out_h: int):
    job_dir = JOBS_DIR / job_id
    temp_video = str(job_dir / "video_only.mp4")
    output_path = str(job_dir / "output.mp4")

    def update(status, progress, message):
        with jobs_lock:
            jobs[job_id].update({"status": status, "progress": progress, "message": message})

    with job_semaphore:
        try:
            ratio = RATIO_MAP.get(ratio_str, 9 / 16)
            out_w = round(out_h * ratio)
            out_w += out_w % 2
            out_h += out_h % 2
            ema_alpha = min(1.0, 3.0 / max(1, ema_smooth))
            analysis_fps = 6

            cap = cv2.VideoCapture(input_path)
            vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            duration = total_frames / input_fps

            pose_w = min(vid_w, 1280)
            pose_h = round(pose_w * vid_h / vid_w)

            update("analyzing", 0, "포즈 분석 중...")

            mp_pose = mp.solutions.pose.Pose(
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            total_analysis = max(1, int(duration * analysis_fps))
            key_times = []
            smoothed_keys = []  # EMA-smoothed + constrained crop boxes
            person_keys = []    # padded person bounds per keyframe

            ema = None

            def fallback_box():
                cw = float(vid_h) * ratio
                ch = float(vid_h)
                if cw > vid_w:
                    cw = float(vid_w)
                    ch = cw / ratio
                return {
                    "sx": (vid_w - cw) / 2, "sy": (vid_h - ch) / 2, "cw": cw, "ch": ch,
                    "px1": 0.0, "px2": float(vid_w), "py1": 0.0, "py2": float(vid_h),
                }

            cap = cv2.VideoCapture(input_path)
            for ai in range(total_analysis):
                t = min(ai / analysis_fps, duration - 0.01)
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ret, frame = cap.read()
                if not ret:
                    break

                small = cv2.resize(frame, (pose_w, pose_h), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                results = mp_pose.process(rgb)

                lm = results.pose_landmarks.landmark if results.pose_landmarks else None
                box = get_crop_box(lm, vid_w, vid_h, ratio, padding) if lm else fallback_box()

                crop = {k: box[k] for k in ("sx", "sy", "cw", "ch")}
                ema = {k: ema[k] + ema_alpha * (crop[k] - ema[k]) for k in crop} if ema else dict(crop)

                # Constrain so person is never cut off even when EMA lags
                sx, sy = constrain_to_person(
                    ema["sx"], ema["sy"], ema["cw"], ema["ch"],
                    box["px1"], box["py1"], box["px2"], box["py2"],
                    vid_w, vid_h,
                )
                smoothed_keys.append({"sx": sx, "sy": sy, "cw": ema["cw"], "ch": ema["ch"]})
                person_keys.append({
                    "sx": box["px1"], "sy": box["py1"],
                    "cw": box["px2"] - box["px1"], "ch": box["py2"] - box["py1"],
                })
                key_times.append(t)

                progress = (ai + 1) / total_analysis * 45
                update("analyzing", progress, f"포즈 분석 중... {progress:.0f}%")

            cap.release()
            mp_pose.close()

            if not key_times:
                raise RuntimeError("영상에서 프레임을 읽을 수 없습니다")

            update("encoding", 45, "인코딩 중...")

            pixels = out_w * out_h
            bitrate = "15M" if pixels <= 2_073_600 else ("25M" if pixels <= 3_686_400 else "45M")

            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{out_w}x{out_h}",
                "-r", str(input_fps),
                "-i", "pipe:0",
                "-c:v", "libx264", "-threads", "0",
                "-preset", "medium", "-b:v", bitrate,
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                temp_video,
            ]
            proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

            cap = cv2.VideoCapture(input_path)
            frame_idx = 0
            total_enc = max(1, int(duration * input_fps))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                t = frame_idx / input_fps
                b = get_box_at(t, key_times, smoothed_keys)
                pb = get_box_at(t, key_times, person_keys)

                # Re-apply constraint on interpolated box
                sx, sy = constrain_to_person(
                    b["sx"], b["sy"], b["cw"], b["ch"],
                    pb["sx"], pb["sy"],
                    pb["sx"] + pb["cw"], pb["sy"] + pb["ch"],
                    vid_w, vid_h,
                )
                cw = min(int(round(b["cw"])), vid_w - int(sx))
                ch = min(int(round(b["ch"])), vid_h - int(sy))

                cropped = frame[int(sy):int(sy) + ch, int(sx):int(sx) + cw]
                resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
                proc.stdin.write(resized.tobytes())
                frame_idx += 1
                if frame_idx % 30 == 0:
                    p = 45 + min(frame_idx / total_enc, 1.0) * 45
                    update("encoding", p, f"인코딩 중... {p:.0f}%")

            cap.release()
            proc.stdin.close()
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError("FFmpeg 인코딩 실패")

            update("encoding", 92, "오디오 합성 중...")
            mux = subprocess.run([
                "ffmpeg", "-y",
                "-i", temp_video,
                "-i", input_path,
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v:0", "-map", "1:a:0?",
                "-shortest", output_path,
            ], stderr=subprocess.DEVNULL)
            if mux.returncode != 0:
                os.rename(temp_video, output_path)
            else:
                if os.path.exists(temp_video):
                    os.unlink(temp_video)

            os.unlink(input_path)
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            update("done", 100, f"완료! ({size_mb:.1f} MB)")
            with jobs_lock:
                jobs[job_id]["output_path"] = output_path

        except Exception:
            tb = traceback.format_exc()
            update("error", 0, "오류: " + tb.splitlines()[-1])


@app.post("/process")
async def start_process(
    file: UploadFile = File(...),
    ratio: str = Form("9:16"),
    padding: float = Form(0.1),
    smooth: int = Form(10),
    out_h: int = Form(1080),
):
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)

    ext = Path(file.filename).suffix or ".mp4"
    input_path = str(job_dir / f"input{ext}")

    with open(input_path, "wb") as f:
        while chunk := await file.read(8 * 1024 * 1024):
            f.write(chunk)

    with jobs_lock:
        jobs[job_id] = {"status": "queued", "progress": 0, "message": "대기 중...", "output_path": None}

    t = threading.Thread(
        target=process_job,
        args=(job_id, input_path, ratio, padding, smooth, out_h),
        daemon=True,
    )
    t.start()
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"status": job["status"], "progress": job["progress"], "message": job["message"]}


@app.get("/download/{job_id}")
def download(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job["status"] != "done" or not job["output_path"]:
        return JSONResponse({"error": "not ready"}, status_code=404)
    output_path = job["output_path"]

    def cleanup():
        import time
        time.sleep(300)
        job_dir = JOBS_DIR / job_id
        shutil.rmtree(job_dir, ignore_errors=True)
        with jobs_lock:
            jobs.pop(job_id, None)

    threading.Thread(target=cleanup, daemon=True).start()
    return FileResponse(output_path, media_type="video/mp4", filename="cropped.mp4")


@app.get("/health")
def health():
    return {"ok": True}
