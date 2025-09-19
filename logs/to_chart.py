import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np


def log_filter(log_path):
    # Open the file lazily, line by line
    with open(log_path, "r") as file:
        for line in file:
            match line.strip().split():
                case [d, t, "TICK"]:
                    dt = datetime.strptime(" ".join([d, t]), "%Y-%m-%d %H:%M:%S,%f")
                    yield dt.timestamp()
                case [
                    _,
                    _,
                    mem_proc,
                    mem_sys,
                ] if mem_proc.isdigit() and mem_sys.isdigit():
                    yield (
                        int(mem_proc) / (1024**3),
                        int(mem_sys) / (1024**3),
                    )  # bytes -> GB


def sma_percentile(y, *, window_size, top=True, percentile=50):
    # Pad y to handle edges (optional, depending on your preference)
    y_padded = np.pad(
        y, (window_size // 2, window_size - 1 - window_size // 2), mode="edge"
    )

    if percentile == 50:
        weights = np.ones(window_size) / window_size
        sma = np.convolve(y_padded, weights, mode="valid")
        return sma

    # Compute moving top 10% average
    sma = np.zeros(len(y))
    for i in range(len(y)):
        window = y_padded[i : i + window_size]
        threshold = np.percentile(window, percentile)
        values = window[window >= threshold] if top else window[window <= threshold]
        sma[i] = np.mean(values) if len(values) > 0 else np.nan
    return sma


def generate_benchmark_charts(log_path: Path, inverse_freq: bool):
    chart_path = log_path.parent / f"benchmark_{log_path.stem}.png"

    ticks = []
    timestamps_sec = []
    mems_proc = []
    mems_sys = []
    for data in log_filter(log_path):
        match data:
            case float() as timestamp:
                ticks.append(timestamp)
                timestamps_sec.append(int(timestamp))  # -> second
            case (float() as mem_proc, float() as mem_sys):
                mems_proc.append(mem_proc)
                mems_sys.append(mem_sys)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    if not inverse_freq:
        # Count frequencies of timestamps in each second using Counter
        frequencies = Counter(timestamps_sec)

        start_time = min(timestamps_sec)
        end_time = max(timestamps_sec)
        dense_range = range(start_time, end_time + 1)

        # Ensure all seconds in the range have a frequency (even if zero)
        dense_frequencies = {sec: frequencies.get(sec, 0) for sec in dense_range}

        x = list(dense_frequencies.keys())  # Timestamps (x-axis)
        y = np.array(list(dense_frequencies.values()))  # Frequency counts (y-axis)
        mean = y.mean()
        std = y.std()

        axs[0].plot(x, y)
        axs[0].plot(
            x[2:],
            sma_percentile(y[2:], window_size=40),
            label="SMA",
            color="orange",
            linewidth=1,
        )
        axs[0].plot(
            x[2:],
            sma_percentile(y[2:], window_size=40, top=True, percentile=90),
            label="SMA Top 10%",
            color="green",
            linewidth=1,
        )
        axs[0].plot(
            x[2:],
            sma_percentile(y[2:], window_size=40, top=False, percentile=10),
            label="SMA Bottom 10%",
            color="red",
            linewidth=1,
        )

        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Freq (hz)")

        axs[0].legend()
    else:
        its = list(range(1, len(ticks)))
        delta_times = np.array([ticks[i] - ticks[i - 1] for i in its])
        mean = delta_times.mean()
        std = delta_times.std()

        axs[0].plot(its[2:], delta_times[2:])
        axs[0].set_xlabel("Iterations")
        axs[0].set_ylabel("Time (s)")
    axs[0].set_title(f"Speed (μ={mean:.2f}, σ={std:.4f})")
    axs[0].grid(True)

    mems_sys = np.array(mems_sys)
    mems_proc = np.array(mems_proc)
    mean_sys = mems_sys.mean()
    std_sys = mems_sys.std()
    mean_proc = mems_proc.mean()
    std_proc = mems_proc.std()

    # tick and memory report are called on different line
    # when ctrl+c, it can cause the one of them to not report at the end at the exact moment it was ctrl+c'd
    mem_min_size = min(len(ticks), mems_sys.shape[0], mems_proc.shape[0])

    axs[1].plot(ticks[:mem_min_size], mems_sys[:mem_min_size], label="System")
    axs[1].plot(ticks[:mem_min_size], mems_proc[:mem_min_size], label="Current Process")
    axs[1].set_ylim(ymin=0)
    axs[1].set_title(
        f"Memory Usage (Sys: μ={mean_sys:.2f}, σ={std_sys:.4f} | Proc: μ={mean_proc:.2f}, σ={std_proc:.4f})"
    )
    axs[1].set_xlabel("Ticks (s)")
    axs[1].set_ylabel("GB")
    axs[1].grid(True)

    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Graph generation tool for benchmarking."
    )
    # add unnamed argument for camera source
    parser.add_argument(
        "path",
        type=Path,
        help="Path to *.log files",
    )
    parser.add_argument("--inverse", dest="inverse", action="store_true")
    parser.set_defaults(inverse=False)
    args = parser.parse_args()

    generate_benchmark_charts(log_path=args.path, inverse_freq=args.inverse)
