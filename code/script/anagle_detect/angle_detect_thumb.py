# angle_detect_thumb.py
# Thumb joint (IP) angle detection from top-view video.
# - Detect beep for time offset
# - Detect colored markers: A=red (MCP), B=blue (IP), D=yellow (tip)
# - Compute IP angle as ∠ABD (signed -> flexion)
# - Unwrap, smooth, and optionally align with motor CSV (thumb motors: 1/4)

import os
import cv2
import numpy as np
import pandas as pd
import librosa
from moviepy.video.io.VideoFileClip import VideoFileClip

# ====================== USER CONFIG ======================
VIDEO_PATH = r"data/video/top/thu2_20250812_181203.mp4"
CSV_PATH   = r"data/raw/flex_sensor/thu2/motor_flex_log_20250812_172157.csv" 

OUTPUT_DIR = "output/thumb_angles"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color thresholds
HSV_RANGES = {
    "red_hi":   ((170,  70,  50), (180, 255, 255)),  
    "red_lo":   ((  0,  70,  50), ( 10, 255, 255)),
    "blue":     ((100,  50,  50), (130, 255, 255)),
    "yellow":   (( 20, 100, 100), ( 35, 255, 255)),
}

# Morphology and thresholds
KERNEL_OPEN  = (3, 3)
KERNEL_CLOSE = (5, 5)
MIN_AREA_PX  = 30

# Smoothing and alignment
SMOOTH_WINDOW = 11   
SMOOTH_POLY   = 2
ASOF_TOLERANCE_MS = 100

# ====================== Utilities ======================
def savgol_maybe(x, window=SMOOTH_WINDOW, poly=SMOOTH_POLY):
    """Savitzky–Golay; if not applicable or sequence too short, return as-is (with NaN fill)."""
    x = np.asarray(x, dtype=float)
    mask = ~np.isfinite(x)
    if mask.all():
        return x
    x_filled = x.copy()
    valid = np.where(~mask)[0]
    for i in range(len(x_filled)):
        if not np.isfinite(x_filled[i]):
            j = valid[np.argmin(np.abs(valid - i))]
            x_filled[i] = x_filled[j]
    if len(x_filled) >= window and window % 2 == 1:
        try:
            from scipy.signal import savgol_filter
            x_s = savgol_filter(x_filled, window_length=window, polyorder=poly)
        except Exception:
            x_s = x_filled
    else:
        x_s = x_filled
    x_s[mask] = np.nan
    return x_s

def detect_beep_time_ms(video_path: str) -> int:
    """Return the time (ms) of the first beep; 0 if failed."""
    audio_path = os.path.join(OUTPUT_DIR, "audio_extracted.wav")
    try:
        with VideoFileClip(video_path) as clip:
            if clip.audio is None:
                print("[WARN] No audio. offset=0 ms.")
                return 0
            clip.audio.write_audiofile(audio_path, fps=clip.audio.fps, logger=None)
    except Exception as e:
        print(f"[WARN] Audio extract failed: {e}. offset=0 ms.")
        return 0
    y, sr = librosa.load(audio_path, sr=None)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(onset_env, sr=sr)
    frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=False)
    beeps = [int(times[i] * 1000) for i in frames]
    print(f"[INFO] Beep candidates (ms): {beeps}")
    return beeps[0] if beeps else 0

def morph(mask):
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERNEL_OPEN)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERNEL_CLOSE)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
    m = cv2.morphologyEx(m,    cv2.MORPH_CLOSE, k2)
    return m

def detect_center_by_range(hsv, lower, upper):
    mask = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    mask = morph(mask)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = [c for c in cnts if cv2.contourArea(c) >= MIN_AREA_PX]
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

def detect_red_center(hsv):

    c1 = detect_center_by_range(hsv, *HSV_RANGES["red_hi"])
    c2 = detect_center_by_range(hsv, *HSV_RANGES["red_lo"])
    if c1 and c2:
        return c1
    return c1 or c2

def get_markers_A_B_D(frame_bgr):
    """A=red(MCP), B=blue(IP), D=yellow(tip)"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    A = detect_red_center(hsv)
    B = detect_center_by_range(hsv, *HSV_RANGES["blue"])
    D = detect_center_by_range(hsv, *HSV_RANGES["yellow"])
    return A, B, D

# ---------- planar signed angle (deg) ----------
def signed_angle_deg(A, B, C):
    """∠ABC, return -180..180."""
    if None in (A, B, C):
        return None
    a = np.array(A, float); b = np.array(B, float); c = np.array(C, float)
    u = a - b  # BA
    v = c - b  # BC
    dot = float(np.dot(u, v))
    cross_z = float(u[0]*v[1] - u[1]*v[0])
    return float(np.degrees(np.arctan2(cross_z, dot)))

def unwrap_deg(prev, curr):
    if prev is None or curr is None:
        return curr
    d = curr - prev
    if d > 180:  curr -= 360
    if d < -180: curr += 360
    return curr

def to_flex(theta_signed):
    """Map signed angle to flexion angle (extension≈0, flexion increases)."""
    if theta_signed is None:
        return None
    return 180.0 - abs(theta_signed)

# ====================== Video pass ======================
def process_video(video_path, t_offset_ms=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[INFO] FPS={fps:.2f}")

    frames = 0
    t_list, ABD_signed, ABD_flex = [], [], []
    prev_ABD = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        A, B, D = get_markers_A_B_D(frame)
        # Thumb IP angle: ∠ABD
        ang = signed_angle_deg(A, B, D)
        ang = unwrap_deg(prev_ABD, ang) if ang is not None else None
        if ang is not None:
            prev_ABD = ang
        flex = to_flex(ang)

        t_ms = int((frames / fps) * 1000) + int(t_offset_ms)
        t_list.append(t_ms)
        ABD_signed.append(ang)
        ABD_flex.append(flex)
        frames += 1

    cap.release()

    ABD_smooth = savgol_maybe(ABD_flex, window=SMOOTH_WINDOW, poly=SMOOTH_POLY)

    df = pd.DataFrame({
        "Frame": np.arange(len(t_list), dtype=int),
        "Time(ms)": t_list,
        "ABD_signed(deg)": ABD_signed,   # Raw signed angle
        "∠ABD": ABD_flex,                # Flexion angle (unsmoothed)
        "∠ABD_smooth": ABD_smooth        # Smoothed flexion angle (for modeling/alignment)
    })
    return df

# ====================== Alignment (optional) ======================
def align_with_motor(angle_df, csv_path, tol_ms=ASOF_TOLERANCE_MS):
    if not csv_path:
        return None
    dfm = pd.read_csv(csv_path, encoding="utf-8-sig")
    dfm = dfm.sort_values("Time(ms)").reset_index(drop=True)
    dfa = angle_df[["Time(ms)", "∠ABD_smooth"]].dropna().sort_values("Time(ms)").reset_index(drop=True)
    merged = pd.merge_asof(dfm, dfa, on="Time(ms)", direction="nearest", tolerance=tol_ms)
    return merged

# ====================== Main ======================
if __name__ == "__main__":
    #Beep alignment
    t0 = detect_beep_time_ms(VIDEO_PATH)
    print(f"[INFO] Using beep offset: {t0} ms")

    #Video processing → Thumb IP angle
    angles = process_video(VIDEO_PATH, t_offset_ms=t0)
    out_angles = os.path.join(OUTPUT_DIR, "angles_thumb.csv")
    angles.to_csv(out_angles, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved: {out_angles}")

    # Align with motor/sensor log
    if CSV_PATH:
        merged = align_with_motor(angles, CSV_PATH, tol_ms=ASOF_TOLERANCE_MS)
        out_aligned = os.path.join(OUTPUT_DIR, "aligned_data_thumb.csv")
        merged.to_csv(out_aligned, index=False, encoding="utf-8-sig")
        print(f"[INFO] Saved: {out_aligned}")
    else:
        print("[INFO] Skip motor alignment (CSV_PATH empty).")
