# anagle_detect_middle.py
# Detect beep offset (robust: bandpass+envelope+peaks), extract colored markers
# (A=red MCP, B=green PIP, C=blue DIP, D=yellow tip), compute PIP=∠ABC and
# DIP=∠BCD (flexion), smooth, and (optionally) align with motor CSV.

import os
import cv2
import numpy as np
import pandas as pd
import librosa
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.signal import savgol_filter, butter, filtfilt, find_peaks

# ====================== USER CONFIG ======================

VIDEO_PATH = r"data/video/top/mid2_20250812_165905.mp4"
CSV_PATH   = r"data/raw/flex_sensor/mid2/motor_flex_log_20250812_163051.csv"
# VIDEO_PATH = r"data/video/top/mid1_20250812_165251.mp4"
# CSV_PATH   = r"data/raw/flex_sensor/mid1/motor_flex_log_20250812_162041.csv"

# Output directory
OUTPUT_DIR = "output/middle_angles"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color thresholds (OpenCV HSV, H:0-180）
HSV_RANGES = {
    "red":    ((170,  70,  50), (180, 255, 255)),
    "green":  (( 35,  50,  50), ( 85, 255, 255)),
    "blue":   ((100,  50,  50), (130, 255, 255)),
    "yellow": (( 20, 100, 100), ( 35, 255, 255)),
}

# Morphology, area, smoothing, alignment parameters
KERNEL_OPEN  = (3, 3)
KERNEL_CLOSE = (5, 5)
MIN_AREA_PX  = 30

SMOOTH_WINDOW = 11
SMOOTH_POLY   = 2

ASOF_TOLERANCE_MS = 100  # Time-nearest tolerance for aligning video angles with motor CSV

# ====================== Beep detection params ======================
BEEP_BAND_HZ = (1000, 8000)     # Bandpass range
PEAK_MIN_DISTANCE_MS = 180      # Minimum distance between peaks, avoid double counting
# FIRST_WINDOW_MS = 5000          # If ≥3 beeps within first 5s, use 3rd beep as Flex start
FIRST_WINDOW_MS = 0
PEAK_SNR_DB = 10                # Peak vs background “approximate SNR” threshold (larger = more conservative)
MANUAL_T0_MS = None             # Manual fallback

# ====================== UTILS ======================
def _bandpass(y: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    nyq = 0.5 * sr
    lo_n = max(lo / nyq, 1e-6)
    hi_n = min(hi / nyq, 0.999999)
    b, a = butter(4, [lo_n, hi_n], btype='band')
    return filtfilt(b, a, y)

def _envelope(x: np.ndarray, sr: int, win_ms: int = 15) -> np.ndarray:
    win = max(1, int(sr * win_ms / 1000))
    x = np.abs(x)
    ker = np.ones(win, dtype=float) / win
    return np.convolve(x, ker, mode='same')

def detect_beep_time_ms(video_path: str) -> int:
    """
    Return beep time (ms) for alignment.
      1) If MANUAL_T0_MS is set, return it directly.
      2) Extract audio from video -> bandpass(1-8k) -> envelope -> find_peaks.
      3) If ≥3 peaks within FIRST_WINDOW_MS, take the 3rd as Flex start; otherwise take the 1st.
      4) Return 0 if failed.
    Note: audio will be exported to OUTPUT_DIR/audio_extracted.wav for debugging;
          librosa onset results will be printed for comparison (not used for alignment).
    """
    if MANUAL_T0_MS is not None:
        print(f"[INFO] Using MANUAL t0: {MANUAL_T0_MS} ms")
        return int(MANUAL_T0_MS)

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

    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        print(f"[WARN] Audio load failed: {e}. offset=0 ms.")
        return 0

    # Robust beep detection: bandpass + envelope + peaks
    y_bp = _bandpass(y, sr, BEEP_BAND_HZ[0], BEEP_BAND_HZ[1])
    env = _envelope(y_bp, sr, win_ms=15)

    med = float(np.median(env))
    mad = float(np.median(np.abs(env - med))) + 1e-9
    alpha = max(6.0, PEAK_SNR_DB / 2.0)
    height = med + alpha * mad

    min_dist = int(sr * (PEAK_MIN_DISTANCE_MS / 1000.0))
    peaks, _ = find_peaks(env, height=height, distance=min_dist)

    if len(peaks) == 0:
        # Relax threshold once and retry
        height = med + 3.0 * mad
        peaks, _ = find_peaks(env, height=height, distance=min_dist)

    if len(peaks) == 0:
        print("[WARN] No peaks found by bandpass+envelope. offset=0 ms.")

        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            times = librosa.times_like(onset_env, sr=sr)
            frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=False)
            beeps_ref = [int(times[i] * 1000) for i in frames]
            print(f"[DBG] librosa onset (reference): {beeps_ref}")
        except Exception:
            pass
        return 0


    t_ms = (peaks / sr) * 1000.0
    t_ms = t_ms.astype(int).tolist()
    print(f"[INFO] Bandpass+peaks beep candidates (ms): {t_ms[:12]}{'...' if len(t_ms)>12 else ''}")

    within = [t for t in t_ms if t <= FIRST_WINDOW_MS]
    if len(within) >= 3:
        chosen = int(within[2])
        print(f"[INFO] Using 3rd peak within first {FIRST_WINDOW_MS} ms as FLEX start: {chosen} ms")
    else:
        chosen = int(t_ms[0])
        print(f"[INFO] Using first peak as offset: {chosen} ms")


    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        times = librosa.times_like(onset_env, sr=sr)
        frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=False)
        beeps_ref = [int(times[i] * 1000) for i in frames]
        print(f"[DBG] librosa onset (NOT used): {beeps_ref}")
    except Exception:
        pass

    return chosen

def morph(mask: np.ndarray):
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERNEL_OPEN)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERNEL_CLOSE)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
    m = cv2.morphologyEx(m,    cv2.MORPH_CLOSE, k2)
    return m

def detect_center(hsv, lower, upper):
    mask = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    mask = morph(mask)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    cnts = [c for c in cnts if cv2.contourArea(c) >= MIN_AREA_PX]
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0: return None
    return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

def get_ABCD(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    A = detect_center(hsv, *HSV_RANGES["red"])
    B = detect_center(hsv, *HSV_RANGES["green"])
    C = detect_center(hsv, *HSV_RANGES["blue"])
    D = detect_center(hsv, *HSV_RANGES["yellow"])
    return A, B, C, D

def angle_signed(A,B,C):
    if None in (A,B,C): return None
    a,b,c = np.array(A,float), np.array(B,float), np.array(C,float)
    u, v = a-b, c-b
    dot = float(np.dot(u,v))
    cross_z = float(u[0]*v[1] - u[1]*v[0])
    return float(np.degrees(np.arctan2(cross_z, dot)))  # -180..180

def unwrap(prev, curr):
    if prev is None or curr is None: return curr
    d = curr - prev
    if d > 180:  curr -= 360
    if d < -180: curr += 360
    return curr

def to_flex(theta_signed):
    if theta_signed is None: return None
    return 180.0 - abs(theta_signed)

def smooth(x, window=SMOOTH_WINDOW, poly=SMOOTH_POLY):
    x = np.asarray(x, float)
    mask = ~np.isfinite(x)
    if mask.all(): return x
    x_f = x.copy()
    valid = np.where(~mask)[0]
    for i in range(len(x_f)):
        if not np.isfinite(x_f[i]):
            j = valid[np.argmin(np.abs(valid - i))]
            x_f[i] = x_f[j]
    if len(x_f) >= window and window % 2 == 1:
        try:
            x_s = savgol_filter(x_f, window_length=window, polyorder=poly)
        except Exception:
            x_s = x_f
    else:
        x_s = x_f
    x_s[mask] = np.nan
    return x_s

# ====================== VIDEO PASS ======================
def process_video(video_path, t_offset_ms=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 1e-3:
        fps = 30.0
    print(f"[INFO] FPS={fps:.2f}")

    t_list, ABC_s, BCD_s, ABC_f, BCD_f = [], [], [], [], []
    prevA = prevD = None

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        A,B,C,D = get_ABCD(frame)

        a_ABC = angle_signed(A,B,C)
        a_BCD = angle_signed(B,C,D)

        a_ABC = unwrap(prevA, a_ABC) if a_ABC is not None else None
        a_BCD = unwrap(prevD, a_BCD) if a_BCD is not None else None
        if a_ABC is not None: prevA = a_ABC
        if a_BCD is not None: prevD = a_BCD

        ABC_s.append(a_ABC); BCD_s.append(a_BCD)
        ABC_f.append(to_flex(a_ABC)); BCD_f.append(to_flex(a_BCD))

        t_ms = int((idx / fps) * 1000) + int(t_offset_ms)
        t_list.append(t_ms)
        idx += 1

    cap.release()

    ABC_sm = smooth(ABC_f)
    BCD_sm = smooth(BCD_f)

    df = pd.DataFrame({
        "Frame": np.arange(len(t_list), dtype=int),
        "Time(ms)": t_list,
        "ABC_signed(deg)": ABC_s,
        "BCD_signed(deg)": BCD_s,
        "∠ABC": ABC_f,
        "∠BCD": BCD_f,
        "∠ABC_smooth": ABC_sm,
        "∠BCD_smooth": BCD_sm
    })
    return df

# ====================== CSV ALIGN (optional) ======================
def align_with_motor(angle_df, csv_path, tol_ms=ASOF_TOLERANCE_MS):
    if not csv_path:
        return None
    dfm = pd.read_csv(csv_path, encoding="utf-8-sig")
    dfm = dfm.sort_values("Time(ms)").reset_index(drop=True)
    dfa = angle_df[["Time(ms)","∠ABC_smooth","∠BCD_smooth"]].copy()
    dfa = dfa.dropna(how="all", subset=["∠ABC_smooth","∠BCD_smooth"]).sort_values("Time(ms)")
    merged = pd.merge_asof(dfm, dfa, on="Time(ms)", direction="nearest", tolerance=tol_ms)
    return merged

# ====================== MAIN ======================
if __name__ == "__main__":
    # 1) Beep offset
    t0 = detect_beep_time_ms(VIDEO_PATH)
    print(f"[INFO] Using beep offset: {t0} ms")

    # 2) Video -> angles
    angles = process_video(VIDEO_PATH, t_offset_ms=t0)
    out_angles = os.path.join(OUTPUT_DIR, "angles_middle.csv")
    angles.to_csv(out_angles, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved: {out_angles}")

    # 3) Optional: align with motor CSV
    if CSV_PATH:
        merged = align_with_motor(angles, CSV_PATH, tol_ms=ASOF_TOLERANCE_MS)
        out_aligned = os.path.join(OUTPUT_DIR, "aligned_data_middle.csv")
        merged.to_csv(out_aligned, index=False, encoding="utf-8-sig")
        print(f"[INFO] Saved: {out_aligned}")
    else:
        print("[INFO] Skip motor alignment (CSV_PATH empty).")
