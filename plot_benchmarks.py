#!/usr/bin/env python3
"""Plot benchmark sweep results: one plot per resolution, kernels on x-axis, grouped by channel count."""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")


def main():
    tsv_file = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.tsv"
    df = pd.read_csv(tsv_file, sep="\t")

    resolutions = sorted(df["resolution"].unique())
    channels = sorted(df["C_in"].unique())

    # Color map for channels
    cmap = plt.cm.viridis
    colors = {ch: cmap(i / max(len(channels) - 1, 1)) for i, ch in enumerate(channels)}

    for res in resolutions:
        df_res = df[df["resolution"] == res]
        kernels = df_res["kernel"].unique()

        fig, ax = plt.subplots(figsize=(max(14, len(kernels) * 0.8), 7))

        x = np.arange(len(kernels))
        bar_width = 0.8 / len(channels)

        for i, ch in enumerate(channels):
            df_ch = df_res[df_res["C_in"] == ch]
            # Align kernel order — fill missing with NaN
            times = []
            for k in kernels:
                rows = df_ch[df_ch["kernel"] == k]
                times.append(rows["time_ms"].values[0] if len(rows) > 0 else np.nan)

            offset = (i - len(channels) / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, times, bar_width, label=f"C={ch}",
                         color=colors[ch], edgecolor="white", linewidth=0.5)

        ax.set_xlabel("Kernel")
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"Conv2D 3x3 Benchmark — {res}x{res} resolution")
        ax.set_xticks(x)
        ax.set_xticklabels(kernels, rotation=45, ha="right", fontsize=8)
        ax.legend(title="Channels (in=out)", fontsize=8, loc="upper left")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        out_path = f"benchmark_{res}x{res}.png"
        fig.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
        plt.close(fig)

    # Also create a TFLOPS plot per resolution
    for res in resolutions:
        df_res = df[df["resolution"] == res]
        kernels = df_res["kernel"].unique()

        fig, ax = plt.subplots(figsize=(max(14, len(kernels) * 0.8), 7))

        x = np.arange(len(kernels))
        bar_width = 0.8 / len(channels)

        for i, ch in enumerate(channels):
            df_ch = df_res[df_res["C_in"] == ch]
            tflops_vals = []
            for k in kernels:
                rows = df_ch[df_ch["kernel"] == k]
                tflops_vals.append(rows["tflops"].values[0] if len(rows) > 0 else np.nan)

            offset = (i - len(channels) / 2 + 0.5) * bar_width
            ax.bar(x + offset, tflops_vals, bar_width, label=f"C={ch}",
                  color=colors[ch], edgecolor="white", linewidth=0.5)

        ax.set_xlabel("Kernel")
        ax.set_ylabel("TFLOPS")
        ax.set_title(f"Conv2D 3x3 Throughput — {res}x{res} resolution")
        ax.set_xticks(x)
        ax.set_xticklabels(kernels, rotation=45, ha="right", fontsize=8)
        ax.legend(title="Channels (in=out)", fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        out_path = f"benchmark_tflops_{res}x{res}.png"
        fig.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
