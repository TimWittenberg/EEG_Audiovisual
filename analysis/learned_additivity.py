"""Learned linear additive models for replay, self-initiated, and attenuation waves.

For each additivity layer, the strict additive model fixes the audio and visual
weights to 1. This script instead *fits* the weights, per subject, by least
squares and compares several reconstructions of the target audiovisual waveform:

    replay      : VAO     = a*AO     + b*VO
    self        : MVA     = a*MA     + b*MV
    attenuation : VAO-MVA = a*(AO-MA) + b*(VO-MV)

For each layer, the compared models are:

    full    : target = a*audio + b*visual     (free audio & visual weights)
    static  : target = 1*audio + 1*visual     (strict additivity)
    audio   : target = a*audio                (audio only)
    visual  : target = b*visual               (visual only)

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
                                           [--model replay|self|attenuation|all]
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


def analyse(experiment: str, site: tuple, model: io.AdditivityModel) -> None:
    folder, domain, channels = site
    cs = io.load_condition_set(experiment, model.conditions)
    out_dir = io.RESULTS_ROOT / folder / model.name
    out_dir.mkdir(parents=True, exist_ok=True)

    audio = model.audio.evaluate(cs)           # (n_subj, n_ch, n_time)
    visual = model.visual.evaluate(cs)
    target = model.target.evaluate(cs)
    if channels is not None:
        roi = cs.roi_indices(channels)
        audio, visual, target = audio[:, roi], visual[:, roi], target[:, roi]
    site_label = "all channels" if channels is None else "+".join(channels)
    n = cs.n_subjects
    comps = io.components_for(domain, experiment)

    # --- Per-subject global fits (pooled over channels x time, no intercept) -
    a = np.zeros(n)
    b = np.zeros(n)
    r2 = {m: np.zeros(n) for m in ("full", "static", "audio", "visual")}

    for i in range(n):
        audio_f = audio[i].ravel()
        visual_f = visual[i].ravel()
        y = target[i].ravel()

        coef_full, r2["full"][i] = _fit(np.column_stack([audio_f, visual_f]), y)
        a[i], b[i] = coef_full
        r2["static"][i] = io.r_squared(y, audio_f + visual_f)
        _, r2["audio"][i] = _fit(audio_f[:, None], y)
        _, r2["visual"][i] = _fit(visual_f[:, None], y)

    # --- Per-subject CSV -----------------------------------------------------
    csv_path = out_dir / f"learned_additivity_{experiment}.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["experiment", "model", "subject", "target_label",
                         "audio_label", "visual_label", "a_audio", "b_visual",
                         "R2_full", "R2_static", "R2_audio", "R2_visual"])
        for i, subject in enumerate(cs.subjects):
            writer.writerow([experiment, model.name, subject, model.target.label,
                             model.audio.label, model.visual.label,
                             f"{a[i]:.5f}", f"{b[i]:.5f}",
                             f"{r2['full'][i]:.5f}", f"{r2['static'][i]:.5f}",
                             f"{r2['audio'][i]:.5f}", f"{r2['visual'][i]:.5f}"])

    # --- Console summary -----------------------------------------------------
    def msd(x: np.ndarray) -> str:
        return f"{x.mean():+.3f} +/- {x.std(ddof=1):.3f}"

    t_ab, p_ab = stats.ttest_rel(a, b)            # audio weight vs visual weight
    print(f"\n=== Learned additive models  {model.formula}   "
          f"[{model.label} | {experiment} | {site_label}] ===")
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

    # --- Time-resolved group fit: target(t) = a(t)*audio + b(t)*visual -------
    # At each time sample, pool all subjects x channels into one regression.
    times = cs.times
    a_t = np.zeros(times.size)
    b_t = np.zeros(times.size)
    r2_t = np.zeros(times.size)
    for k in range(times.size):
        audio_k = audio[:, :, k].ravel()
        visual_k = visual[:, :, k].ravel()
        y_k = target[:, :, k].ravel()
        coef, r2_t[k] = _fit(np.column_stack([audio_k, visual_k]), y_k)
        a_t[k], b_t[k] = coef

    tr_path = out_dir / f"learned_time_resolved_{experiment}.csv"
    with tr_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["experiment", "model", "time_ms", "a_audio", "b_visual", "R2"])
        for k in range(times.size):
            writer.writerow([experiment, model.name, f"{times[k] * 1e3:.2f}",
                             f"{a_t[k]:.5f}", f"{b_t[k]:.5f}", f"{r2_t[k]:.5f}"])
    print(f"written: {tr_path.relative_to(io.PROJECT_ROOT)}")

    # --- Figure --------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f"Learned additive models  {model.formula}  -  {experiment}  "
                 f"({model.label}; {site_label}, n={n})")

    # Panel 1: per-subject weights -- the jittered per-subject cloud, with the
    # group mean and +/- SEM drawn as a marker centred on each cloud (no bars:
    # the raw points already carry the distribution).
    ax = axes[0]
    ax.axhline(1.0, color="0.5", ls="--", lw=1, label="strict additivity (=1)")
    ax.axhline(0.0, color="0.8", lw=0.8)
    ax.scatter(np.zeros(n) + np.random.uniform(-0.08, 0.08, n), a,
               color="tab:blue", s=18, alpha=0.6, zorder=3)
    ax.scatter(np.ones(n) + np.random.uniform(-0.08, 0.08, n), b,
               color="tab:green", s=18, alpha=0.6, zorder=3)
    for x0, vals in ((0, a), (1, b)):
        ax.errorbar(x0, vals.mean(), yerr=vals.std(ddof=1) / np.sqrt(n),
                    color="k", marker="o", markersize=5, capsize=4,
                    lw=1.5, zorder=5)

    # Significance bracket for the a-vs-b paired t-test (the audio-dominance test).
    top = max(a.max(), b.max())
    y_br = top + 0.10
    bar_h = 0.03 * max(1.0, top)
    ax.plot([0, 0, 1, 1], [y_br, y_br + bar_h, y_br + bar_h, y_br],
            color="k", lw=1.2)
    sig = ("***" if p_ab < 0.001 else "**" if p_ab < 0.01
           else "*" if p_ab < 0.05 else "n.s.")
    ax.text(0.5, y_br + bar_h,
            f"audio vs visual: {sig}  (paired t = {t_ab:+.2f}, p = {p_ab:.3f})",
            ha="center", va="bottom", fontsize=8)
    ax.set_ylim(top=y_br + bar_h + 0.22 * max(1.0, top))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["a (audio)", "b (visual)"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Fitted weight")
    ax.set_title("Per-subject weights  (mean +/- SEM marker on the cloud)")
    ax.legend(loc="lower right", fontsize=8)

    # Panel 2: model comparison by R^2. Tick labels carry the weight constraint
    # that defines each model, so the panel reads without the docstring.
    ax = axes[1]
    models = ("full", "static", "audio", "visual")
    labels = ("full\n(a, b free)", "static\n(a=b=1)",
              "audio\n(b=0)", "visual\n(a=0)")
    means = [r2[m].mean() for m in models]
    sems = [r2[m].std(ddof=1) / np.sqrt(n) for m in models]
    ax.bar(labels, means, yerr=sems, capsize=5,
           color=["tab:purple", "0.6", "tab:blue", "tab:green"], alpha=0.7)
    ax.set_ylabel(f"R^2 (reconstruction of {model.target.label})")
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
    parser.add_argument("--model", choices=(*io.ADDITIVITY_MODELS, "all"), default="all")
    args = parser.parse_args()
    np.random.seed(0)  # only affects jitter in the scatter plot
    experiments = io.EXPERIMENTS if args.exp == "both" else (args.exp,)
    models = (io.ADDITIVITY_MODELS.values() if args.model == "all"
              else (io.ADDITIVITY_MODELS[args.model],))
    for experiment in experiments:
        for model in models:
            for site in SITES:
                analyse(experiment, site, model)


if __name__ == "__main__":
    main()
