# anagle_detect_index.py
# Detect beep (for time offset), extract color markers from top-view video,
# compute SIGNED joint angles with atan2, unwrap & smooth to get flexion angles,
# load serial CSV, and align motor & angle data by time.

import os
import cv2
import numpy as np
import pandas as pd
import librosa
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.signal import savgol_filter, butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

# ====================== CONFIG ======================
VIDEO_PATH = r"data/video/top/ind2_20250812_164915.mp4"
CSV_PATH   = r"data/raw/flex_sensor/ind2/motor_flex_log_20250812_160704.csv"
# VIDEO_PATH = r"data/video/top/ind1_20250812_164558.mp4"
# CSV_PATH   = r"data/raw/flex_sensor/ind1/motor_flex_log_20250812_160008.csv"

# HSV ranges for colored stickers (H:0-180 in OpenCV)
HSV_RANGES = {
    "red":    ((170,  70,  50), (180, 255, 255)),
    "green":  (( 35,  50,  50), ( 85, 255, 255)),
    "blue":   ((100,  50,  50), (130, 255, 255)),
    "yellow": (( 20, 100, 100), ( 35, 255, 255)),
}

# morphology & contour filter
KERNEL_OPEN  = (3, 3)   # opening to remove speckles
KERNEL_CLOSE = (5, 5)   # closing to fill gaps
MIN_AREA_PX  = 30       # minimal contour area to accept

# smoothing
SMOOTH_WINDOW = 11       # odd number; if sequence length is insufficient it will automatically skip
SMOOTH_POLY   = 2

# alignment
ASOF_TOLERANCE_MS = 100     # nearest-neighbor tolerance for time alignment

# output
OUTPUT_DIR = "output/index_angles"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================== Beep detection params ======================
# —— New robust beep detection (bandpass + envelope + peaks), with rule “3rd beep within first 5s = Flex start” ——
BEEP_BAND_HZ = (1000, 8000)     # Beeps usually at several kHz, open wider for stability
PEAK_MIN_DISTANCE_MS = 180      # Minimum interval between two beeps (avoid counting one long beep twice)
# FIRST_WINDOW_MS = 5000          # If ≥3 beeps detected within first 5s, use 3rd beep as Flex start
FIRST_WINDOW_MS = 0
PEAK_SNR_DB = 10                # Minimum “approximate SNR” requirement of peak relative to background (larger = more conservative)
MANUAL_T0_MS = None             # Manual fallback (fill specific ms when needed; None = not used)


# ====================== Utilities ======================
def _bandpass(y: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    """IIR bandpass filter (Butterworth, 4th order)."""
    nyq = 0.5 * sr
    lo_n = max(lo / nyq, 1e-6)
    hi_n = min(hi / nyq, 0.999999)
    b, a = butter(4, [lo_n, hi_n], btype='band')
    return filtfilt(b, a, y)

def _envelope(x: np.ndarray, sr: int, win_ms: int = 15) -> np.ndarray:
    """Simple envelope: |x| then moving average to suppress fine noise."""
    win = max(1, int(sr * win_ms / 1000))
    x = np.abs(x)
    ker = np.ones(win, dtype=float) / win
    return np.convolve(x, ker, mode='same')

def detect_beep_times(video_path: str) -> int:
    """
    Return beep time (ms) for alignment.
    Logic:
      1) If MANUAL_T0_MS is set, return it directly (manual fallback).
      2) Extract audio from video -> bandpass(1-8k) -> envelope -> find_peaks.
      3) If ≥3 peaks detected within FIRST_WINDOW_MS (default 5s), take the 3rd as Flex start;
         otherwise use the 1st peak.
      4) Return 0 if failed.
    Note:
      - Save temporary audio to OUTPUT_DIR/audio_extracted.wav (for debugging).
      - Keep librosa onset result printed for reference (not used for alignment), for comparison.
    """
    # manual offset
    if MANUAL_T0_MS is not None:
        print(f"[INFO] Using MANUAL t0: {MANUAL_T0_MS} ms")
        return int(MANUAL_T0_MS)

    audio_path = os.path.join(OUTPUT_DIR, "audio_extracted.wav")
    try:
        with VideoFileClip(video_path) as clip:
            if clip.audio is None:
                print("[WARN] No audio stream found; using 0 ms offset.")
                return 0
            clip.audio.write_audiofile(audio_path, fps=clip.audio.fps, logger=None)
    except Exception as e:
        print(f"[WARN] Failed to extract audio: {e}. Using 0 ms offset.")
        return 0

    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        print(f"[WARN] Failed to load audio: {e}. Using 0 ms offset.")
        return 0

    # bandpass + envelope + peaks
    y_bp = _bandpass(y, sr, BEEP_BAND_HZ[0], BEEP_BAND_HZ[1])
    env = _envelope(y_bp, sr, win_ms=15)

    med = float(np.median(env))
    mad = float(np.median(np.abs(env - med))) + 1e-9
    alpha = max(6.0, PEAK_SNR_DB / 2.0)  
    height = med + alpha * mad

    min_dist = int(sr * (PEAK_MIN_DISTANCE_MS / 1000.0))
    peaks, props = find_peaks(env, height=height, distance=min_dist)

    #  If no peaks found, retry with relaxed threshold
    if len(peaks) == 0:
        height = med + 3.0 * mad
        peaks, props = find_peaks(env, height=height, distance=min_dist)

    if len(peaks) == 0:
        print("[WARN] No peaks found by bandpass+envelope. offset=0 ms.")
        # Print onsets for reference
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            times = librosa.times_like(onset_env, sr=sr)
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=False)
            onset_beeps_ms = [int(times[i] * 1000) for i in onset_frames]
            print(f"[DBG] librosa onset (for reference): {onset_beeps_ms}")
        except Exception:
            pass
        return 0

    # Peak times (ms)
    t_ms = (peaks / sr) * 1000.0
    t_ms = t_ms.astype(int).tolist()
    print(f"[INFO] Bandpass+peaks beep candidates (ms): {t_ms[:12]}{'...' if len(t_ms)>12 else ''}")

    # Rule: if ≥3 peaks within first 5s, use the 3rd as Flex start
    within = [t for t in t_ms if t <= FIRST_WINDOW_MS]
    if len(within) >= 3:
        print(f"[INFO] Using 3rd peak within first {FIRST_WINDOW_MS} ms as FLEX start: {within[2]} ms")
        chosen = int(within[2])
    else:
        chosen = int(t_ms[0])
        print(f"[INFO] Using first peak as offset: {chosen} ms")

    # Also print librosa onset result (debug only, not used for alignment)
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        times = librosa.times_like(onset_env, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=False)
        onset_beeps_ms = [int(times[i] * 1000) for i in onset_frames]
        print(f"[DBG] librosa onset (NOT used for align): {onset_beeps_ms}")
    except Exception:
        pass

    return chosen


def morph_mask(mask: np.ndarray) -> np.ndarray:
    """Apply opening then closing to clean the mask."""
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERNEL_OPEN)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERNEL_CLOSE)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    m = cv2.morphologyEx(m,    cv2.MORPH_CLOSE, k_close)
    return m

def detect_marker_center(hsv_img, lower, upper):
    mask = cv2.inRange(hsv_img, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    mask = morph_mask(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # pick largest reasonable contour
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA_PX]
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def get_markers_from_frame(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    A = detect_marker_center(hsv, *HSV_RANGES["red"])
    B = detect_marker_center(hsv, *HSV_RANGES["green"])
    C = detect_marker_center(hsv, *HSV_RANGES["blue"])
    D = detect_marker_center(hsv, *HSV_RANGES["yellow"])
    return A, B, C, D

# ---------- signed angle in 2D with atan2 ----------
def signed_angle_deg_2d(A, B, C):
    """
    Return signed angle ∠ABC in degrees (-180..180], using 2D cross product sign.
    Positive sign corresponds to ccw rotation from BA to BC in image coordinates.
    """
    if None in (A, B, C):
        return None
    a = np.array(A, dtype=float)
    b = np.array(B, dtype=float)
    c = np.array(C, dtype=float)
    u = a - b  # BA
    v = c - b  # BC
    # dot and 2D "cross" (z-component)
    dot = float(np.dot(u, v))
    cross_z = float(u[0]*v[1] - u[1]*v[0])
    ang = np.degrees(np.arctan2(cross_z, dot))  # -180..180
    return ang

def unwrap_deg(prev_deg, curr_deg):
    """Unwrap current degree near previous."""
    if prev_deg is None or curr_deg is None:
        return curr_deg
    d = curr_deg - prev_deg
    if d > 180:
        curr_deg -= 360
    elif d < -180:
        curr_deg += 360
    return curr_deg

def to_flex_deg(theta_signed):
    """Physiological flexion angle: 0 (extended) -> larger as flex increases."""
    if theta_signed is None:
        return None
    return 180.0 - abs(theta_signed)

def smooth_series(x, window=SMOOTH_WINDOW, poly=SMOOTH_POLY):
    """Savitzky–Golay smoothing for numeric list with Nones handled."""
    x = np.asarray(x, dtype=float)
    # mask Nones/NaNs
    mask = ~np.isfinite(x)
    if mask.all():
        return x
    # simple forward-fill for gaps to allow filtering
    x_filled = x.copy()
    valid_idx = np.where(~mask)[0]
    for i in range(len(x_filled)):
        if not np.isfinite(x_filled[i]):
            # nearest valid
            j = valid_idx[np.argmin(np.abs(valid_idx - i))]
            x_filled[i] = x_filled[j]
    # apply filter if sufficient length & window odd
    if len(x_filled) >= window and window % 2 == 1:
        try:
            x_s = savgol_filter(x_filled, window_length=window, polyorder=poly)
        except Exception:
            x_s = x_filled
    else:
        x_s = x_filled
    x_s[mask] = np.nan
    return x_s

# ====================== Video pass ======================
def process_video(video_path, t_offset_ms=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 1e-3:
        fps = 30.0  # fallback
    print(f"[INFO] Video FPS: {fps:.3f}")

    frame_idx = 0
    t_list = []
    ABC_signed, BCD_signed = [], []
    ABC_flex,   BCD_flex   = [], []

    # running unwrap
    prev_ABC = None
    prev_BCD = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        A, B, C, D = get_markers_from_frame(frame)

        ang_ABC = signed_angle_deg_2d(A, B, C)
        ang_BCD = signed_angle_deg_2d(B, C, D)

        # unwrap for continuity
        ang_ABC = unwrap_deg(prev_ABC, ang_ABC) if ang_ABC is not None else None
        ang_BCD = unwrap_deg(prev_BCD, ang_BCD) if ang_BCD is not None else None

        prev_ABC = ang_ABC if ang_ABC is not None else prev_ABC
        prev_BCD = ang_BCD if ang_BCD is not None else prev_BCD

        # flexion angle (0 extended -> larger with flexion)
        flex_ABC = to_flex_deg(ang_ABC)
        flex_BCD = to_flex_deg(ang_BCD)

        timestamp = int((frame_idx / fps) * 1000) + int(t_offset_ms)

        t_list.append(timestamp)
        ABC_signed.append(ang_ABC)
        BCD_signed.append(ang_BCD)
        ABC_flex.append(flex_ABC)
        BCD_flex.append(flex_BCD)

        frame_idx += 1

    cap.release()

    # smoothing
    ABC_flex_s = smooth_series(ABC_flex)
    BCD_flex_s = smooth_series(BCD_flex)

    df = pd.DataFrame({
        "Frame": np.arange(len(t_list), dtype=int),
        "Time(ms)": t_list,
        "ABC_signed(deg)": ABC_signed,
        "BCD_signed(deg)": BCD_signed,
        "∠ABC": ABC_flex,                 # raw flexion
        "∠BCD": BCD_flex,
        "∠ABC_smooth": ABC_flex_s,        # smoothed flexion
        "∠BCD_smooth": BCD_flex_s
    })
    return df

# ====================== Load motor CSV ======================
def load_motor_csv(csv_path, motor_ids=(1,2,3,4,5,6)):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    # basic sort
    df = df.sort_values(["Time(ms)", "MotorID"]).reset_index(drop=True)
    if motor_ids is not None:
        df = df[df["MotorID"].isin(list(motor_ids))].copy()
    return df

# ====================== Align by time ======================
def merge_data(angle_df, motor_df, use_smoothed=True, tol_ms=ASOF_TOLERANCE_MS):
    # choose which angle columns to use for downstream modeling
    a_cols = ["∠ABC_smooth", "∠BCD_smooth"] if use_smoothed else ["∠ABC", "∠BCD"]
    keep_cols = ["Time(ms)"] + a_cols

    angle_df2 = angle_df[keep_cols].dropna(subset=a_cols, how="all").copy()
    angle_df2 = angle_df2.sort_values("Time(ms)").reset_index(drop=True)

    motor_df2 = motor_df.sort_values("Time(ms)").reset_index(drop=True)

    merged = pd.merge_asof(
        motor_df2, angle_df2,
        on="Time(ms)",
        direction="nearest",
        tolerance=tol_ms
    )
    return merged

# ====================== Main ======================
if __name__ == "__main__":
    # Step 1: beep for time offset
    t0 = detect_beep_times(VIDEO_PATH)
    print(f"\n[INFO] Final time offset used for alignment: {t0} ms")

    # Step 2: process video -> angles
    print("[INFO] Processing video for angles (signed + flex + smooth)...")
    angle_df = process_video(VIDEO_PATH, t_offset_ms=t0)
    angle_csv = os.path.join(OUTPUT_DIR, "angles_index.csv")
    angle_df.to_csv(angle_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved angles to: {angle_csv}")

    # Step 3: load motor data
    print("[INFO] Loading motor CSV...")
    motor_df = load_motor_csv(CSV_PATH)
    motor_csv = os.path.join(OUTPUT_DIR, "motor_data_filtered.csv")
    motor_df.to_csv(motor_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved motor subset to: {motor_csv}")

    # Step 4: align & export
    print("[INFO] Aligning by time (nearest, tolerance={} ms)...".format(ASOF_TOLERANCE_MS))
    merged = merge_data(angle_df, motor_df, use_smoothed=True, tol_ms=ASOF_TOLERANCE_MS)
    out_csv = os.path.join(OUTPUT_DIR, "aligned_data_index.csv")
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] All results saved to: {OUTPUT_DIR}")
