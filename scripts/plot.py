import os
import matplotlib.pyplot as plt
import numpy as np

log_dir = "logs"
log_file_path = os.path.join(log_dir, "manager.log")

frame_batch_size = 500
window_size = 10
outlier_thresh = 3000

x = []
y = []

# Parse log
with open(log_file_path, "r") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            _, reward_str, _ = line.split(", ")
            reward = float(reward_str)
            x.append(i * frame_batch_size)
            y.append(reward)
        except Exception:
            print(f"Skipping malformed line: {line}")

# smooth non-outliers
y_smoothed = []
for i in range(len(y)):
    if y[i] >= outlier_thresh:
        y_smoothed.append(y[i])
    else:
        # rolling average from neighbors
        start = max(0, i - window_size // 2)
        end = min(len(y), i + window_size // 2 + 1)
        window = y[start:end]
        # exclude outliers in the smoothing window
        filtered_window = [val for val in window if val < outlier_thresh]
        if filtered_window:
            smoothed_val = sum(filtered_window) / len(filtered_window)
        else:
            smoothed_val = y[i]
        y_smoothed.append(smoothed_val)


min_y = min(y_smoothed)
if min_y <= 0:
    y_smoothed = [val - min_y + 1e-5 for val in y_smoothed]

plt.figure(figsize=(12, 6))
plt.plot(x, y_smoothed, label="Smoothed (non-outliers)", linewidth=2)
plt.yscale("log")
plt.title("Smoothed Reward Over Frames (Outliers Preserved)")
plt.xlabel("Frames")
plt.ylabel("Reward (log scale)")
ticks = [300, 1000, 7000]
tick_labels = [str(t) for t in ticks]

plt.yticks(ticks, tick_labels)



plt.axhline(y=4300, color='red', linestyle='--', linewidth=1.5, label='Level Completion Threshold')


# roughly at 1.8M frames is the 12H mark
plt.axvline(x=1_800_000, color='blue', linestyle='--', linewidth=1.5, label='12H mark')
plt.axvline(x=3_750_000, color='purple', linestyle='--', linewidth=1.5, label='24H mark')
plt.axvline(x=5_400_000, color='orange', linestyle='--', linewidth=1.5, label='36H mark')

# first win
x_position = 2_550_000
y_position = 4600
plt.annotate("First win",
             xy=(x_position, y_position), 
             xytext=(x_position * 1.05, y_position * 1.1),
             arrowprops=dict(arrowstyle="->", color='green'),
             fontsize=10, color='green')

text_x = 3_080_000
text_y = 3600

arrow_pos1 = (text_x * 1.08, text_y * .8)
arrow_pos2 = (text_x * 0.99, text_y * .7)
plt.annotate("Near wins",
             xy=(arrow_pos1[0], arrow_pos1[1]),
             xytext=(text_x, text_y),
             arrowprops=dict(arrowstyle="->", color='orange'),
                fontsize=10, color='orange')
plt.annotate("Near wins",
             xy=(arrow_pos2[0], arrow_pos2[1]),
             xytext=(text_x, text_y),
             arrowprops=dict(arrowstyle="->", color='orange'),
                fontsize=10, color='orange')


#plt.text(2_500_000, 5000, "First win", fontsize=10, color='green')

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("reward_plot.png", dpi=300, bbox_inches='tight')
plt.show()
