"""Static additive-model tests for replay, self-initiated, and attenuation waves.

For every subject this computes a difference wave for one additivity layer:

    replay      : D(t) = VAO(t)     - [ AO(t)    + VO(t)    ]
    self        : D(t) = MVA(t)     - [ MA(t)    + MV(t)    ]
    attenuation : D(t) = VAO-MVA(t) - [ AO-MA(t) + VO-MV(t) ]

The first two layers test additivity of the sensory response itself. The
attenuation layer directly tests additivity of the self-vs-external effect. A
reliable departure of D from zero is evidence against strict additivity for the
chosen layer. The script summarises D across the group and reports
component-level deviations in the predefined component windows.

READ-ONLY on the source data. Output goes to results/.

Usage:
    python analysis/static_additivity.py [--exp Exp1|Exp2|both]
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

UV = 1e6  # volts -> microvolts for reporting/plotting

# Reporting sites. Each is (subfolder, domain, channels): the channels are
# averaged into one waveform; the domain selects the ERP components
# (io.components_for). Each site's output goes in its own results/ subfolder.
SITES = (("ROI", "visual", io.ROI_CHANNELS), ("Cz", "auditory", ("Cz",)))


def analyse(experiment: str, site: tuple, model: io.AdditivityModel) -> None:
    folder, domain, channels = site
    cs = io.load_condition_set(experiment, model.conditions)
    out_dir = io.RESULTS_ROOT / folder / model.name
    out_dir.mkdir(parents=True, exist_ok=True)

    audio = model.audio.evaluate(cs)           # (n_subj, n_ch, n_time)
    visual = model.visual.evaluate(cs)
    target = model.target.evaluate(cs)
    predicted = audio + visual                 # strict additive prediction
    diff = target - predicted                  # multisensory interaction wave

    times = cs.times
    roi = cs.roi_indices(channels)
    roi_label = "+".join(channels)
    comps = io.components_for(domain, experiment)

    # ROI-averaged waveforms, one row per subject -> (n_subj, n_time).
    target_roi = target[:, roi].mean(axis=1)
    pred_roi = predicted[:, roi].mean(axis=1)
    diff_roi = diff[:, roi].mean(axis=1)

    # Whole-epoch reconstruction quality per subject, on the site waveform(s).
    target_site = target[:, roi]
    pred_site = predicted[:, roi]
    per_subject_r2 = np.array(
        [io.r_squared(target_site[i], pred_site[i]) for i in range(cs.n_subjects)]
    )
    per_subject_rmse = np.sqrt(
        np.mean((target_site - pred_site).reshape(cs.n_subjects, -1) ** 2, axis=1)
    ) * UV

    # Component-level deviations (ROI mean amplitude in each window).
    rows = []
    comp_dev = {}
    for name, spec in comps.items():
        window = spec["window"]
        target_c = io.window_mean(target_roi, times, window) * UV
        pred_c = io.window_mean(pred_roi, times, window) * UV
        d_c = target_c - pred_c
        t_val, p_val = stats.ttest_1samp(d_c, 0.0)
        comp_dev[name] = dict(target=target_c, pred=pred_c, dev=d_c, t=t_val, p=p_val)

    for i, subject in enumerate(cs.subjects):
        row = {
            "experiment": experiment,
            "model": model.name,
            "subject": subject,
            "target_label": model.target.label,
            "audio_label": model.audio.label,
            "visual_label": model.visual.label,
        }
        for name, c in comp_dev.items():
            row[f"target_{name}_uV"] = c["target"][i]
            row[f"audioPlusVisual_{name}_uV"] = c["pred"][i]
            row[f"D_{name}_uV"] = c["dev"][i]
        row["whole_epoch_R2"] = per_subject_r2[i]
        row["whole_epoch_RMSE_uV"] = per_subject_rmse[i]
        rows.append(row)

    csv_path = out_dir / f"static_additivity_{experiment}.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v)
                             for k, v in row.items()})

    # --- Console summary -----------------------------------------------------
    n = cs.n_subjects
    print(f"\n=== Static additive model  {model.formula}   "
          f"[{model.label} | {experiment} | {roi_label}] ===")
    print(f"subjects: {n}   site: {roi_label}")
    if cs.skipped:
        print(f"skipped (incomplete condition set): {cs.skipped}")
    print(f"whole-epoch reconstruction: R2 = {per_subject_r2.mean():.3f} "
          f"+/- {per_subject_r2.std(ddof=1):.3f}   "
          f"RMSE = {per_subject_rmse.mean():.3f} uV")
    for name in comp_dev:
        c = comp_dev[name]
        dev = c["dev"]
        sem = dev.std(ddof=1) / np.sqrt(n)
        print(f"  {name}: target={c['target'].mean():+.3f}  "
              f"audio+visual={c['pred'].mean():+.3f}  "
              f"D={dev.mean():+.3f} +/- {sem:.3f} uV  "
              f"t({n - 1})={c['t']:+.2f}  p={c['p']:.4f}")
    print(f"written: {csv_path.relative_to(io.PROJECT_ROOT)}")

    # --- Figure --------------------------------------------------------------
    t_ms = times * 1e3
    grand_diff = diff_roi.mean(axis=0) * UV
    diff_sem = diff_roi.std(axis=0, ddof=1) / np.sqrt(n) * UV
    _, p_per_t = stats.ttest_1samp(diff_roi, 0.0, axis=0)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle(f"Static additive model  {model.formula}  -  {experiment}  "
                 f"({model.label}; {roi_label}, n={n})")

    ax_top.plot(t_ms, target_roi.mean(0) * UV, color="tab:blue", lw=2,
                label=f"{model.target.label} (observed)")
    ax_top.plot(t_ms, pred_roi.mean(0) * UV, color="tab:red", lw=2, ls="--",
                label=f"{model.audio.label} + {model.visual.label} (predicted)")
    ax_top.set_ylabel("Amplitude (uV)")
    ax_top.legend(loc="upper right")
    ax_top.set_title("Grand-average waveforms")

    ax_bot.axhline(0, color="0.6", lw=0.8)
    ax_bot.plot(t_ms, grand_diff, color="tab:purple", lw=2,
                label=f"D = {model.target.label} - "
                      f"({model.audio.label}+{model.visual.label})")
    ax_bot.fill_between(t_ms, grand_diff - diff_sem, grand_diff + diff_sem,
                        color="tab:purple", alpha=0.25, label="+/- SEM")
    sig = p_per_t < 0.05
    if sig.any():
        y_sig = np.full(t_ms.shape, np.nan)
        y_sig[sig] = ax_bot.get_ylim()[0]
        ax_bot.plot(t_ms, y_sig, color="k", lw=4, solid_capstyle="butt",
                    label="p < 0.05 (uncorrected)")
    ax_bot.set_ylabel("Difference (uV)")
    ax_bot.set_xlabel("Time relative to stimulus (ms)")
    ax_bot.set_title("Multisensory interaction (difference wave)")
    ax_bot.legend(loc="upper right")

    for ax in (ax_top, ax_bot):
        ax.axvline(0, color="0.6", lw=0.8)
        for (name, spec), col in zip(comps.items(), ("tab:green", "tab:orange")):
            win = spec["window"]
            ax.axvspan(win[0] * 1e3, win[1] * 1e3, color=col, alpha=0.10)

    fig.tight_layout()
    fig_path = out_dir / f"static_additivity_{experiment}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"written: {fig_path.relative_to(io.PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", choices=(*io.EXPERIMENTS, "both"), default="both")
    parser.add_argument("--model", choices=(*io.ADDITIVITY_MODELS, "all"), default="all")
    args = parser.parse_args()
    experiments = io.EXPERIMENTS if args.exp == "both" else (args.exp,)
    models = (io.ADDITIVITY_MODELS.values() if args.model == "all"
              else (io.ADDITIVITY_MODELS[args.model],))
    for experiment in experiments:
        for model in models:
            for site in SITES:
                analyse(experiment, site, model)


if __name__ == "__main__":
    main()
