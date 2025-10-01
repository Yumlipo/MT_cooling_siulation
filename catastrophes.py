import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from itertools import groupby

# Your data
# lengths = np.array([1, 2, 3, 4, 3.5, 3, 2, 1, 0.5, 0, 1, 2, 3, 2, 1, 0])

def catastrophe_calc(lengths, fig_name):

    # Smooth data (reduce noise)
    window_size = 500
    smoothed = savgol_filter(lengths, window_size, 3)

    # Compute slope (1st derivative)
    slope = np.gradient(smoothed)

    # Thresholds for trend detection
    # increase_threshold = 0.15
    decrease_threshold = -1.1

    # Classify trends
    trend = np.zeros_like(slope)
    # trend[slope > increase_threshold] = 1    # Increasing
    trend[slope < decrease_threshold] = -1   # Decreasing

    # Find continuous segments
    segments = []
    for key, group in groupby(enumerate(trend), key=lambda x: x[1]):
        if key != 0:
            indices = [i for i, val in group]
            start, end = indices[0], indices[-1]
            
            # if len(segments) < 4 or start > segments[-3] + 500:
            segments.append((start, end, key))
            print("start", segments[-1][-3])
            print("end", segments[-1][-2])

    # Filter short fluctuations
    min_segment_length = 3
    filtered_segments = [(s, e, t) for s, e, t in segments if (e - s + 1) >= min_segment_length]

    # Plotting
    plt.figure(figsize=(10, 6))

    # 1. Original and smoothed data
    plt.plot(lengths, 'o-', linewidth=3, label='Original Data', alpha=0.5)
    plt.plot(smoothed, 'b-', linewidth=2, label='Smoothed Data')

    # 2. Highlight trends
    colors = {1: 'green', -1: 'red'}
    for start, end, trend_dir in filtered_segments:
        plt.axvspan(start, end, alpha=0.2, color=colors[trend_dir], 
                    label=f'{"Increasing" if trend_dir == 1 else "Decreasing"} Trend')

    # 3. Slope (derivative) plot
    plt.twinx()
    plt.plot(slope, 'k--', label='Slope (Derivative)')
    # plt.axhline(increase_threshold, color='green', linestyle=':', alpha=0.5, label='Increase Threshold')
    plt.axhline(decrease_threshold, color='red', linestyle=':', alpha=0.5, label='Decrease Threshold')
    plt.axhline(0, color='gray', linestyle='-', alpha=0.5)

    # Labels and legend
    plt.title('Trend Detection in Data (Ignoring Small Fluctuations)')
    plt.xlabel('Index')
    plt.ylabel('Value / Slope')
    plt.legend(loc='upper left')
    plt.grid(True)

    plt.savefig(fig_name)

    # plt.show()
    return len(filtered_segments)
