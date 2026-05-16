"""Learned linear additive models for the replay conditions.

The strict additive model assumes VAO = 1*AO + 1*VO. This script instead
*fits* the weights, per subject, by least squares and compares several
reconstructions of the audiovisual response:

    full    : VAO = a*AO + b*VO          (free audio & visual weights)
    static  : VAO = 1*AO + 1*VO          (fixed reference; strict additivity)
    audio   : VAO = a*AO                 (audio only)
    visual  : VAO = b*VO                 (visual only)

No intercept is fitted: the strict additive model is exactly (a, b) = (1, 1),
so a free offset would blur that interpretation. The data are baseline-
corrected, so there is no DC offset for an intercept to absorb anyway.

The audio weight a vs. the visual weight b, and the R^2 of the reduced
audio-only / visual-only models, quantify how far audio dominates the
audiovisual response. A third, time-resolved fit shows whether that dominance
is concentrated around the N1.

READ-ONLY on the source data. Output goes to results/.

Usage:
    python analysis/learned_additivity.py [--exp Exp1|Exp2|both]
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
from scipy import stats

import io_utils as io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

UV = 1e6

# Reporting sites. (subfolder, domain, channels): the channels are averaged
# into one waveform; the domain selects the ERP components (io.components_for).
SITES = (("ROI", "visual", io.ROI_CHANNELS), ("Cz", "auditory", ("Cz",)))


def _fit(design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares fit; returns (coefficients, R^2)."""
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coef, io.r_squared(target, design @ coef)


def analyse(experiment: str, site: tuple) -> None:
    folder, domain, channels = site
    cs = io.load_condition_set(experiment, io.REPLAY_CONDITIONS)
    out_dir = io.RESULTS_ROOT / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    ao = cs.cond(io.AUDITORY_CONDITION)        # (n_subj, n_ch, n_time)
    vo = cs.cond(io.VISUAL_CONDITION)
    vao = cs.cond(io.AUDIOVISUAL_CONDITION)
    if channels is not None:
        roi = cs.roi_indices(channels)
        ao, vo, vao = ao[:, roi], vo[:, roi], vao[:, roi]
    site_label = "all channels" if channels is None else "+".join(channels)
    n = cs.n_subjects
    comps = io.components_for(domain, experiment)

    # --- Per-subject global fits (pooled over channels x time, no intercept) -
    a = np.zeros(n)
    b = np.zeros(n)
    r2 = {m: np.zeros(n) for m in ("full", "static", "audio", "visual")}

    for i in range(n):
        ao_f = ao[i].ravel()
        vo_f = vo[i].ravel()
        y = vao[i].ravel()

        coef_full, r2["full"][i] = _fit(np.column_stack([ao_f, vo_f]), y)
        a[i], b[i] = coef_full
        r2["static"][i] = io.r_squared(y, ao_f + vo_f)
        _, r2["audio"][i] = _fit(ao_f[:, None], y)
        _, r2["visual"][i] = _fit(vo_f[:, None], y)

    # --- Per-subject CSV -----------------------------------------------------
    csv_path = out_dir / f"learned_additivity_{experiment}.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["experiment", "subject", "a_audio", "b_visual",
                         "R2_full", "R2_static", "R2_audio", "R2_visual"])
        for i, subject in enumerate(cs.subjects):
            writer.writerow([experiment, subject,
                             f"{a[i]:.5f}", f"{b[i]:.5f}",
                             f"{r2['full'][i]:.5f}", f"{r2['static'][i]:.5f}",
                             f"{r2['audio'][i]:.5f}", f"{r2['visual'][i]:.5f}"])

    # --- Console summary -----------------------------------------------------
    def msd(x: np.ndarray) -> str:
        return f"{x.mean():+.3f} +/- {x.std(ddof=1):.3f}"

    t_ab, p_ab = stats.ttest_rel(a, b)            # audio weight vs visual weight
    print(f"\n=== Learned additive models  [{experiment} | {site_label}] ===")
    print(f"subjects: {n}")
    if cs.skipped:
        print(f"skipped (incomplete condition set): {cs.skipped}")
    print(f"  audio weight  a : {msd(a)}")
    print(f"  visual weight b : {msd(b)}")
    print(f"  a vs b (paired t-test): t({n - 1})={t_ab:+.2f}  p={p_ab:.4f}  "
          f"-> {'audio > visual' if a.mean() > b.mean() else 'visual >= audio'}")
    print("  mean R^2 by model:")
    for m in ("full", "static", "audio", "visual"):
        print(f"    {m:7s}: {r2[m].mean():.3f} +/- {r2[m].std(ddof=1):.3f}")
    print(f"written: {csv_path.relative_to(io.PROJECT_ROOT)}")

    # --- Time-resolved group fit  VAO(t) = a(t)*AO + b(t)*VO -----------------
    # At each time sample, pool all subjects x channels into one regression.
    times = cs.times
    a_t = np.zeros(times.size)
    b_t = np.zeros(times.size)
    r2_t = np.zeros(times.size)
    for k in range(times.size):
        ao_k = ao[:, :, k].ravel()
        vo_k = vo[:, :, k].ravel()
        y_k = vao[:, :, k].ravel()
        coef, r2_t[k] = _fit(np.column_stack([ao_k, vo_k]), y_k)
        a_t[k], b_t[k] = coef

    tr_path = out_dir / f"learned_time_resolved_{experiment}.csv"
    with tr_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["experiment", "time_ms", "a_audio", "b_visual", "R2"])
        for k in range(times.size):
            writer.writerow([experiment, f"{times[k] * 1e3:.2f}",
                             f"{a_t[k]:.5f}", f"{b_t[k]:.5f}", f"{r2_t[k]:.5f}"])
    print(f"written: {tr_path.relative_to(io.PROJECT_ROOT)}")

    # --- Figure --------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f"Learned additive models  -  {experiment}  ({site_label}, n={n})")

    # Panel 1: per-subject weights.
    ax = axes[0]
    ax.bar([0, 1], [a.mean(), b.mean()],
           yerr=[a.std(ddof=1) / np.sqrt(n), b.std(ddof=1) / np.sqrt(n)],
           color=["tab:blue", "tab:green"], capsize=5, alpha=0.7)
    ax.scatter(np.zeros(n) + np.random.uniform(-0.08, 0.08, n), a,
               color="tab:blue", s=18, zorder=3)
    ax.scatter(np.ones(n) + np.random.uniform(-0.08, 0.08, n), b,
               color="tab:green", s=18, zorder=3)
    ax.axhline(1.0, color="0.5", ls="--", lw=1, label="strict additivity (=1)")

    # Significance bracket for the a-vs-b paired t-test.
    top = max(a.max(), b.max())
    y_br = top + 0.10
    bar_h = 0.03 * max(1.0, top)
    ax.plot([0, 0, 1, 1], [y_br, y_br + bar_h, y_br + bar_h, y_br],
            color="k", lw=1.2)
    sig = ("***" if p_ab < 0.001 else "**" if p_ab < 0.01
           else "*" if p_ab < 0.05 else "n.s.")
    ax.text(0.5, y_br + bar_h,
            f"{sig}   paired t({n - 1}) = {t_ab:+.2f},  p = {p_ab:.3f}",
            ha="center", va="bottom", fontsize=8)
    ax.set_ylim(top=y_br + bar_h + 0.22 * max(1.0, top))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["a (audio)", "b (visual)"])
    ax.set_ylabel("Fitted weight")
    ax.set_title("Per-subject weights")
    ax.legend(loc="lower right", fontsize=8)

    # Panel 2: model comparison by R^2.
    ax = axes[1]
    models = ("full", "static", "audio", "visual")
    means = [r2[m].mean() for m in models]
    sems = [r2[m].std(ddof=1) / np.sqrt(n) for m in models]
    ax.bar(models, means, yerr=sems, capsize=5,
           color=["tab:purple", "0.6", "tab:blue", "tab:green"], alpha=0.7)
    ax.set_ylabel("R^2 (reconstruction of VAO)")
    ax.set_title("Model comparison")
    ax.set_ylim(min(0.0, min(means) - 0.1), 1.0)

    # Panel 3: time-resolved weights.
    ax = axes[2]
    t_ms = times * 1e3
    ax.axhline(1.0, color="0.5", ls="--", lw=1)
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.plot(t_ms, a_t, color="tab:blue", lw=2, label="a(t) audio")
    ax.plot(t_ms, b_t, color="tab:green", lw=2, label="b(t) visual")
    ax.axvline(0, color="0.6", lw=0.8)
    for (name, spec), col in zip(comps.items(), ("tab:green", "tab:orange")):
        win = spec["window"]
        ax.axvspan(win[0] * 1e3, win[1] * 1e3, color=col, alpha=0.10)
    ax.set_xlabel("Time relative to stimulus (ms)")
    ax.set_ylabel("Fitted weight")
    ax.set_title("Time-resolved weights")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig_path = out_dir / f"learned_additivity_{experiment}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"written: {fig_path.relative_to(io.PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", choices=(*io.EXPERIMENTS, "both"), default="both")
    args = parser.parse_args()
    np.random.seed(0)  # only affects jitter in the scatter plot
    experiments = io.EXPERIMENTS if args.exp == "both" else (args.exp,)
    for experiment in experiments:
        for site in SITES:
            analyse(experiment, site)


if __name__ == "__main__":
    main()
