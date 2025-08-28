# read_serial.py : Extract the output content from the serial port, and then organize, save and visualize the data.

import serial
import pandas as pd
import matplotlib.pyplot as plt

# === Configuration ===
PORT = 'COM13'          
BAUDRATE = 115200          
CSV_FILE = 'motor_data_log.csv'
MOTOR_IDS = [1, 2, 3, 4, 5, 6]

# === Initialize Serial Port ===
ser = serial.Serial(PORT, BAUDRATE, timeout=1)

# === Data Storage ===
timestamps = []
motor_ids = []
phases = []
directions = []
positions = []
angles = []

# === Real-Time Plotting Setup ===
plt.ion()
fig, ax = plt.subplots()

lines = {}
data_buffer = {}
colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan']

for idx, motor_id in enumerate(MOTOR_IDS):
    lines[motor_id], = ax.plot([], [], label=f'Motor {motor_id}', color=colors[idx])
    data_buffer[motor_id] = {'x': [], 'y': []}

ax.set_xlabel('Time (ms)')
ax.set_ylabel('Angle (°)')
ax.set_title('Live Motor Angle')
ax.legend()
plt.show()

try:
    while True:
        line_raw = ser.readline().decode(errors='ignore').strip()
        if not line_raw or line_raw.startswith('===') or line_raw.startswith('Time'):
            continue  # Skip header or debug lines

        print(line_raw)  

        parts = line_raw.split(',')
        if len(parts) != 6 or not parts[0].isdigit():
            continue  # Skip invalid format

        try:
            t = int(parts[0])
            m_id = int(parts[1])
            phase = parts[2]
            direction = parts[3]
            pos = int(parts[4])
            angle = float(parts[5])
        except ValueError:
            continue  # Skip malformed data

        # Save to full list
        timestamps.append(t)
        motor_ids.append(m_id)
        phases.append(phase)
        directions.append(direction)
        positions.append(pos)
        angles.append(angle)

        if m_id in MOTOR_IDS:
            data_buffer[m_id]['x'].append(t)
            data_buffer[m_id]['y'].append(angle)

        # Real-time plot update
        for m_id in MOTOR_IDS:
            lines[m_id].set_xdata(data_buffer[m_id]['x'])
            lines[m_id].set_ydata(data_buffer[m_id]['y'])

        ax.relim()
        ax.autoscale_view()
        plt.pause(0.001)

except KeyboardInterrupt:
    print("\n[INFO] KeyboardInterrupt received. Saving data...")

finally:
    ser.close()
    plt.ioff()

    # Save full CSV
    df = pd.DataFrame({
        'Time(ms)': timestamps,
        'MotorID': motor_ids,
        'Phase': phases,
        'Direction': directions,
        'Position': positions,
        'Angle(deg)': angles
    })
    df.to_csv(CSV_FILE, index=False)
    print(f"[INFO] Full log saved to {CSV_FILE}")

    # Save each motor to its own CSV
    for m_id in MOTOR_IDS:
        motor_df = df[df['MotorID'] == m_id]
        motor_df.to_csv(f'motor_{m_id}_log.csv', index=False)
        print(f"[INFO] Saved motor {m_id} to motor_{m_id}_log.csv")

    # Plot each motor in a subplot
    fig, axs = plt.subplots(len(MOTOR_IDS), 1, figsize=(10, 12), sharex=True)
    fig.suptitle("Live Motor Angles (Each Motor in Subplot)")

    for idx, m_id in enumerate(MOTOR_IDS):
        ax = axs[idx]
        ax.plot(data_buffer[m_id]['x'], data_buffer[m_id]['y'], color=colors[idx])
        ax.set_ylabel(f"M{m_id} Angle (°)")
        ax.grid(True)

    axs[-1].set_xlabel("Time (ms)")
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
