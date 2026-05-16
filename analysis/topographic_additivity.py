"""Per-channel additive fit and scalp topography of the audio/visual weights.

The global learned fit pools all 32 channels into one regression, so
high-amplitude channels dominate the estimated weights. This script instead
fits the additive model

    VAO(t) = a*AO(t) + b*VO(t)        (no intercept; strict additivity is a=b=1)

INDEPENDENTLY for every channel and every subject, then averages across
subjects. Every channel is weighted equally, and the result is a scalp map of
a (audio weight), b (visual weight) and the difference a-b (audio dominance).

READ-ONLY on the source data. Output goes to results/.

Usage:
    python analysis/topographic_additivity.py [--exp Exp1|Exp2|both]
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

import io_utils as io

import mne  # noqa: E402  (imported after io_utils so MPLCONFIGDIR is set)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

mne.set_log_level("ERROR")


def fit_channel(ao: np.ndarray, vo: np.ndarray, vao: np.ndarray) -> tuple[float, float, float]:
    """No-intercept least-squares fit  VAO = a*AO + b*VO  for one channel."""
    design = np.column_stack([ao, vo])
    coef, *_ = np.linalg.lstsq(design, vao, rcond=None)
    a, b = coef
    return float(a), float(b), io.r_squared(vao, design @ coef)


def make_info(ch_names: list[str], sfreq: float) -> mne.Info:
    """An MNE Info with a standard 10-20 montage for topographic plotting."""
    info = mne.create_info(list(ch_names), sfreq, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, match_case=False, on_missing="warn")
    return info


def analyse(experiment: str) -> None:
    cs = io.load_condition_set(experiment, io.REPLAY_CONDITIONS)
    io.RESULTS_ROOT.mkdir(exist_ok=True)

    ao = cs.cond(io.AUDITORY_CONDITION)        # (n_subj, n_ch, n_time)
    vo = cs.cond(io.VISUAL_CONDITION)
    vao = cs.cond(io.AUDIOVISUAL_CONDITION)
    n, n_ch, _ = ao.shape

    # Fit every (subject, channel) independently.
    a = np.zeros((n, n_ch))
    b = np.zeros((n, n_ch))
    r2 = np.zeros((n, n_ch))
    for i in range(n):
        for ch in range(n_ch):
            a[i, ch], b[i, ch], r2[i, ch] = fit_channel(ao[i, ch], vo[i, ch], vao[i, ch])

    # Average across subjects -> one value per channel.
    a_mean, b_mean, r2_mean = a.mean(0), b.mean(0), r2.mean(0)
    a_sem = a.std(0, ddof=1) / np.sqrt(n)
    b_sem = b.std(0, ddof=1) / np.sqrt(n)
    dominance = a_mean - b_mean                 # audio dominance per channel

    # --- Per-channel CSV -----------------------------------------------------
    csv_path = io.RESULTS_ROOT / f"topographic_additivity_{experiment}.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["experiment", "channel", "a_audio", "a_sem",
                         "b_visual", "b_sem", "R2", "audio_minus_visual"])
        for ch, name in enumerate(cs.ch_names):
            writer.writerow([experiment, name,
                             f"{a_mean[ch]:.5f}", f"{a_sem[ch]:.5f}",
                             f"{b_mean[ch]:.5f}", f"{b_sem[ch]:.5f}",
                             f"{r2_mean[ch]:.5f}", f"{dominance[ch]:.5f}"])

    # --- Console summary -----------------------------------------------------
    print(f"\n=== Per-channel additive fit  VAO = a*AO + b*VO   [{experiment}] ===")
    print(f"subjects: {n}   channels: {n_ch}")
    if cs.skipped:
        print(f"skipped (incomplete condition set): {cs.skipped}")
    print(f"  mean over channels: a={a_mean.mean():+.3f}  b={b_mean.mean():+.3f}  "
          f"R2={r2_mean.mean():.3f}")
    order = np.argsort(dominance)[::-1]
    print("  strongest audio dominance (a - b):")
    for ch in order[:5]:
        print(f"    {cs.ch_names[ch]:>4s}: a={a_mean[ch]:+.3f}  b={b_mean[ch]:+.3f}  "
              f"a-b={dominance[ch]:+.3f}")
    print("  weakest audio dominance:")
    for ch in order[-3:]:
        print(f"    {cs.ch_names[ch]:>4s}: a={a_mean[ch]:+.3f}  b={b_mean[ch]:+.3f}  "
              f"a-b={dominance[ch]:+.3f}")
    print(f"written: {csv_path.relative_to(io.PROJECT_ROOT)}")

    # --- Topographic figure --------------------------------------------------
    info = make_info(cs.ch_names, cs.sfreq)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3))
    fig.suptitle(f"Per-channel additive fit  VAO = a*AO + b*VO  -  {experiment}  "
                 f"(n={n})")

    # Shared colour scale for a and b so the two weight maps are comparable.
    wlo = min(a_mean.min(), b_mean.min())
    whi = max(a_mean.max(), b_mean.max())
    dmax = float(np.max(np.abs(dominance)))

    panels = [
        ("a  (audio weight)", a_mean, "viridis", (wlo, whi)),
        ("b  (visual weight)", b_mean, "viridis", (wlo, whi)),
        ("a - b  (audio dominance)", dominance, "RdBu_r", (-dmax, dmax)),
        ("R^2  (fit quality)", r2_mean, "viridis", (0.0, 1.0)),
    ]
    for ax, (title, values, cmap, vlim) in zip(axes, panels):
        im, _ = mne.viz.plot_topomap(values, info, axes=ax, cmap=cmap,
                                     vlim=vlim, contours=6, sensors=True,
                                     show=False)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.7)

    fig.tight_layout()
    fig_path = io.RESULTS_ROOT / f"topographic_additivity_{experiment}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"written: {fig_path.relative_to(io.PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", choices=(*io.EXPERIMENTS, "both"), default="both")
    args = parser.parse_args()
    experiments = io.EXPERIMENTS if args.exp == "both" else (args.exp,)
    for experiment in experiments:
        analyse(experiment)


if __name__ == "__main__":
    main()
