# read_serial2.py : Read Arduino serial, live plot (rolling), and save FULL-session CSVs & plots.

import os
import time
import serial
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime

# =============== Configuration ===============
PORT = 'COM3'
BAUDRATE = 115200

SAVE_DIR = r'C:\Users\nicey\Desktop\Soft_glove\data\raw\flex_sensor'
MOTOR_IDS = [1, 2, 3, 4, 5, 6]

# Live plotting keeps only the latest N points
LIVE_MAX_POINTS_MOTOR  = 3000
LIVE_MAX_POINTS_SENSOR = 3000

#  For final save plots: maximum number of sampled points
FINAL_PLOT_MAX_SAMPLES = 200000

plt.ion()

# =============== Prepare paths ===============
os.makedirs(SAVE_DIR, exist_ok=True)
ts_label = datetime.now().strftime('%Y%m%d_%H%M%S')

base_csv        = os.path.join(SAVE_DIR, f'motor_flex_log_{ts_label}.csv')
sensors_csv     = os.path.join(SAVE_DIR, f'flex_sensors_{ts_label}.csv')
plot_motors_png = os.path.join(SAVE_DIR, f'plot_motors_{ts_label}.png')         # 实时窗口图（可被覆盖）
plot_sensors_png= os.path.join(SAVE_DIR, f'plot_sensors_{ts_label}.png')        # 实时窗口图（可被覆盖）
full_motors_png = os.path.join(SAVE_DIR, f'plot_motors_full_{ts_label}.png')    # 全程图
full_sensors_png= os.path.join(SAVE_DIR, f'plot_sensors_full_{ts_label}.png')   # 全程图

print(f'[INFO] Opening serial {PORT} @ {BAUDRATE} ...')
ser = serial.Serial(PORT, BAUDRATE, timeout=1)
time.sleep(2)

# =============== Storage ===============
full_rows = {
    'Time(ms)': [], 'MotorID': [], 'Phase': [], 'Direction': [], 'Position': [], 'Angle(deg)': [],
    'SensorA0': [], 'SensorA1': [], 'SensorA2': []
}

# Rolling live buffer
lines_motors = {}
buf_motor_x = {m: deque(maxlen=LIVE_MAX_POINTS_MOTOR) for m in MOTOR_IDS}
buf_motor_y = {m: deque(maxlen=LIVE_MAX_POINTS_MOTOR) for m in MOTOR_IDS}

lines_sensors = {}
buf_s_t  = deque(maxlen=LIVE_MAX_POINTS_SENSOR)
buf_s_a0 = deque(maxlen=LIVE_MAX_POINTS_SENSOR)
buf_s_a1 = deque(maxlen=LIVE_MAX_POINTS_SENSOR)
buf_s_a2 = deque(maxlen=LIVE_MAX_POINTS_SENSOR)

# =============== Live Plot setup ===============
fig1, ax1 = plt.subplots()
colors = ['tab:blue','tab:red','tab:green','tab:orange','tab:purple','tab:cyan']
for i, m in enumerate(MOTOR_IDS):
    (lines_motors[m],) = ax1.plot([], [], label=f'Motor {m}', linewidth=1.5, color=colors[i % len(colors)])
ax1.set_xlabel('Time (ms)'); ax1.set_ylabel('Angle (°)')
ax1.set_title('Live Motor Angles'); ax1.legend(loc='upper left'); ax1.grid(True)

fig2, ax2 = plt.subplots()
(lines_sensors['A0'],) = ax2.plot([], [], label='Sensor A0', linewidth=1.5)
(lines_sensors['A1'],) = ax2.plot([], [], label='Sensor A1', linewidth=1.5)
(lines_sensors['A2'],) = ax2.plot([], [], label='Sensor A2', linewidth=1.5)
ax2.set_xlabel('Time (ms)'); ax2.set_ylabel('ADC Value')
ax2.set_title('Live Flex Sensors (A0/A1/A2)'); ax2.legend(loc='upper left'); ax2.grid(True)

plt.tight_layout(); plt.show()

def parse_line(line: str):
    parts = [p.strip() for p in line.split(',')]
    if len(parts) != 9:
        return None
    if not parts[0].isdigit():
        return None
    try:
        return {
            'Time(ms)': int(parts[0]),
            'MotorID' : int(parts[1]),
            'Phase'   : parts[2],
            'Direction': parts[3],
            'Position': int(parts[4]),
            'Angle(deg)': float(parts[5]),
            'SensorA0': int(parts[6]),
            'SensorA1': int(parts[7]),
            'SensorA2': int(parts[8]),
        }
    except ValueError:
        return None

def update_live(row):
    t, m = row['Time(ms)'], row['MotorID']
    angle = row['Angle(deg)']
    a0, a1, a2 = row['SensorA0'], row['SensorA1'], row['SensorA2']

    if m in buf_motor_x:
        buf_motor_x[m].append(t); buf_motor_y[m].append(angle)

    buf_s_t.append(t); buf_s_a0.append(a0); buf_s_a1.append(a1); buf_s_a2.append(a2)

    # motors
    for mid in MOTOR_IDS:
        lines_motors[mid].set_xdata(buf_motor_x[mid])
        lines_motors[mid].set_ydata(buf_motor_y[mid])
    ax1.relim(); ax1.autoscale_view()

    # sensors
    lines_sensors['A0'].set_xdata(buf_s_t);  lines_sensors['A0'].set_ydata(buf_s_a0)
    lines_sensors['A1'].set_xdata(buf_s_t);  lines_sensors['A1'].set_ydata(buf_s_a1)
    lines_sensors['A2'].set_xdata(buf_s_t);  lines_sensors['A2'].set_ydata(buf_s_a2)
    ax2.relim(); ax2.autoscale_view()

    plt.pause(0.001)

print('[INFO] Start reading. Ctrl+C to stop.')
try:
    while True:
        raw = ser.readline().decode(errors='ignore').strip()
        if not raw:
            continue
        if raw.startswith('===') or raw.startswith('Time(') or raw.startswith('Time'):
            continue

        row = parse_line(raw)
        if row is None:
            continue

        for k in full_rows:
            full_rows[k].append(row[k])

        update_live(row)

except KeyboardInterrupt:
    print('\n[INFO] KeyboardInterrupt: saving data...')

finally:
    try:
        ser.close()
    except Exception:
        pass

    # ===== Save CSV =====
    df = pd.DataFrame(full_rows)
    df.to_csv(base_csv, index=False, encoding='utf-8-sig')
    print(f'[INFO] Full log saved: {base_csv}')

    # Per-motor CSV
    for m in MOTOR_IDS:
        m_df = df[df['MotorID'] == m]
        m_path = os.path.join(SAVE_DIR, f'motor_{m}_{ts_label}.csv')
        m_df.to_csv(m_path, index=False, encoding='utf-8-sig')
        print(f'[INFO] Saved motor {m}: {m_path}')

    # Sensors-only CSV
    sensors_df = df[['Time(ms)','SensorA0','SensorA1','SensorA2']].drop_duplicates(
        subset=['Time(ms)'], keep='last').sort_values('Time(ms)')
    sensors_df.to_csv(sensors_csv, index=False, encoding='utf-8-sig')
    print(f'[INFO] Saved sensors: {sensors_csv}')

    try:
        fig1.savefig(plot_motors_png, dpi=150)
        fig2.savefig(plot_sensors_png, dpi=150)
        print(f'[INFO] Saved rolling-window plots: {plot_motors_png}, {plot_sensors_png}')
    except Exception as e:
        print(f'[WARN] Save rolling plot failed: {e}')

    # Re-plot FULL SESSION from CSV
    try:
        # 1) Motors full
        fig_full1, ax_full1 = plt.subplots(figsize=(10,6))
        for i, m in enumerate(MOTOR_IDS):
            m_df = df[df['MotorID']==m][['Time(ms)','Angle(deg)']].copy()
            if m_df.empty: continue
            # Downsample if too many points
            if len(m_df) > FINAL_PLOT_MAX_SAMPLES:
                step = len(m_df) // FINAL_PLOT_MAX_SAMPLES + 1
                m_df = m_df.iloc[::step, :]
            ax_full1.plot(m_df['Time(ms)'], m_df['Angle(deg)'],
                          label=f'Motor {m}', linewidth=1.0,
                          color=colors[i % len(colors)])
        ax_full1.set_title('Motor Angles - Full Session')
        ax_full1.set_xlabel('Time (ms)'); ax_full1.set_ylabel('Angle (°)')
        ax_full1.legend(loc='upper left'); ax_full1.grid(True)
        fig_full1.tight_layout(); fig_full1.savefig(full_motors_png, dpi=180)
        plt.close(fig_full1)
        print(f'[INFO] Saved FULL motor plot: {full_motors_png}')

        # 2) Sensors full
        fig_full2, ax_full2 = plt.subplots(figsize=(10,6))
        s_df = sensors_df  # already de-duplicated, sorted
        if len(s_df) > FINAL_PLOT_MAX_SAMPLES:
            step = len(s_df) // FINAL_PLOT_MAX_SAMPLES + 1
            s_df = s_df.iloc[::step, :]
        ax_full2.plot(s_df['Time(ms)'], s_df['SensorA0'], label='Sensor A0', linewidth=1.0)
        ax_full2.plot(s_df['Time(ms)'], s_df['SensorA1'], label='Sensor A1', linewidth=1.0)
        ax_full2.plot(s_df['Time(ms)'], s_df['SensorA2'], label='Sensor A2', linewidth=1.0)
        ax_full2.set_title('Flex Sensors (A0/A1/A2) - Full Session')
        ax_full2.set_xlabel('Time (ms)'); ax_full2.set_ylabel('ADC Value')
        ax_full2.legend(loc='upper left'); ax_full2.grid(True)
        fig_full2.tight_layout(); fig_full2.savefig(full_sensors_png, dpi=180)
        plt.close(fig_full2)
        print(f'[INFO] Saved FULL sensors plot: {full_sensors_png}')

    except Exception as e:
        print(f'[WARN] Failed to save FULL plots: {e}')

    plt.ioff()
    plt.show()
