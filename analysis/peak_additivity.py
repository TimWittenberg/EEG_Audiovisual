"""Peak-based (latency-independent) additivity test for the replay conditions.

The waveform scripts ask whether VAO(t) == AO(t) + VO(t) sample by sample. This
script instead reduces every condition to its ERP-component *peaks* -- the extremum
is picked wherever it falls inside the search window, so individual differences
in peak latency no longer blur the amplitude.

Two prediction definitions are computed side by side, because "peak of a sum"
is not "sum of two peaks" when the peaks occur at different latencies:

    sum-of-peaks (SoP) : predicted = peak(AO) + peak(VO)        [peaks independent]
    peak-of-sum  (PoS) : predicted = peak( AO(t) + VO(t) )      [sum first, then peak]

For each definition it runs both analyses requested:

    static : D = peak(VAO) - predicted            (one-sample t-test vs 0)
    learned: peak(VAO) = a*<audio> + b*<visual> + c   (fit ACROSS subjects)

The learned fit is necessarily group-level: each subject yields a single peak
value per condition, so a, b are estimated across the subject sample (n = 22 /
30), not within a subject as in the waveform script.

It also reports peak *latencies* -- e.g. whether VAO's N1 latency tracks AO's,
which is independent evidence on audio dominance.

READ-ONLY on the source data. Output goes to results/.

Usage:
    python analysis/peak_additivity.py [--exp Exp1|Exp2|both]
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
from scipy import optimize, stats

import io_utils as io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

UV = 1e6  # volts -> microvolts

# Reporting sites. (subfolder, domain, channels): the channels are averaged
# into one waveform; the domain selects the ERP components (io.components_for).
SITES = (("ROI", "visual", io.ROI_CHANNELS), ("Cz", "auditory", ("Cz",)))


def pick_peak(waveform: np.ndarray, times: np.ndarray,
              window: tuple[float, float], polarity: float) -> tuple[float, float]:
    """Return (amplitude, latency_s) of the extremum inside the search window."""
    lo, hi = window
    mask = (times >= lo) & (times <= hi)
    candidates = np.flatnonzero(mask)
    idx = candidates[np.argmax(polarity * waveform[mask])]
    return float(waveform[idx]), float(times[idx])


def fit_lstsq(design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coef, io.r_squared(target, design @ coef)


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def analyse(experiment: str, site: tuple) -> None:
    folder, domain, channels = site
    cs = io.load_condition_set(experiment, io.REPLAY_CONDITIONS)
    out_dir = io.RESULTS_ROOT / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    times = cs.times
    roi = cs.roi_indices(channels)
    roi_label = "+".join(channels)
    n = cs.n_subjects
    comps = io.components_for(domain, experiment)

    # ROI-averaged waveforms per subject, in microvolts.
    ao_w = cs.cond(io.AUDITORY_CONDITION)[:, roi].mean(1) * UV   # (n_subj, n_time)
    vo_w = cs.cond(io.VISUAL_CONDITION)[:, roi].mean(1) * UV
    vao_w = cs.cond(io.AUDIOVISUAL_CONDITION)[:, roi].mean(1) * UV
    sum_w = ao_w + vo_w

    rows: list[dict] = []        # long-format per-subject CSV
    result: dict[str, dict] = {} # per-component metrics for summary + figure

    for comp, spec in comps.items():
        win = spec["window"]
        pol = spec["polarity"]

        ao_amp = np.zeros(n); ao_lat = np.zeros(n)
        vo_amp = np.zeros(n); vo_lat = np.zeros(n)
        vao_amp = np.zeros(n); vao_lat = np.zeros(n)
        pos_pred = np.zeros(n); pos_lat = np.zeros(n)
        for i in range(n):
            ao_amp[i], ao_lat[i] = pick_peak(ao_w[i], times, win, pol)
            vo_amp[i], vo_lat[i] = pick_peak(vo_w[i], times, win, pol)
            vao_amp[i], vao_lat[i] = pick_peak(vao_w[i], times, win, pol)
            pos_pred[i], pos_lat[i] = pick_peak(sum_w[i], times, win, pol)

        sop_pred = ao_amp + vo_amp
        d_sop = vao_amp - sop_pred                 # static, sum-of-peaks
        d_pos = vao_amp - pos_pred                 # static, peak-of-sum
        t_sop, p_sop = stats.ttest_1samp(d_sop, 0.0)
        t_pos, p_pos = stats.ttest_1samp(d_pos, 0.0)

        # Learned, sum-of-peaks: linear regression across subjects.
        coef_sop, r2l_sop = fit_lstsq(
            np.column_stack([ao_amp, vo_amp, np.ones(n)]), vao_amp
        )

        # Learned, peak-of-sum: pick the peak of (a*AO + b*VO), then add offset c.
        # Peak-picking is non-linear, so this is a small numerical fit.
        def residuals(params: np.ndarray, _win=win, _pol=pol) -> np.ndarray:
            a, b, c = params
            pred = np.array([
                pick_peak(a * ao_w[i] + b * vo_w[i], times, _win, _pol)[0]
                for i in range(n)
            ]) + c
            return pred - vao_amp

        fit = optimize.least_squares(residuals, x0=np.array([1.0, 1.0, 0.0]))
        coef_pos = fit.x
        r2l_pos = io.r_squared(vao_amp, vao_amp + fit.fun)

        # Latency: does VAO's peak latency track the audio or the visual peak?
        r_lat_ao = stats.pearsonr(vao_lat, ao_lat)
        r_lat_vo = stats.pearsonr(vao_lat, vo_lat)

        result[comp] = dict(
            ao_amp=ao_amp, vo_amp=vo_amp, vao_amp=vao_amp,
            ao_lat=ao_lat, vo_lat=vo_lat, vao_lat=vao_lat,
            d_sop=d_sop, d_pos=d_pos,
            t_sop=t_sop, p_sop=p_sop, t_pos=t_pos, p_pos=p_pos,
            coef_sop=coef_sop, r2l_sop=r2l_sop,
            coef_pos=coef_pos, r2l_pos=r2l_pos,
            r_lat_ao=r_lat_ao, r_lat_vo=r_lat_vo,
        )

        for i, subject in enumerate(cs.subjects):
            rows.append(dict(
                experiment=experiment, subject=subject, component=comp,
                AO_amp_uV=ao_amp[i], VO_amp_uV=vo_amp[i], VAO_amp_uV=vao_amp[i],
                AO_lat_ms=ao_lat[i] * 1e3, VO_lat_ms=vo_lat[i] * 1e3,
                VAO_lat_ms=vao_lat[i] * 1e3,
                SoP_pred_uV=sop_pred[i], PoS_pred_uV=pos_pred[i],
                D_SoP_uV=d_sop[i], D_PoS_uV=d_pos[i],
            ))

    # --- Per-subject CSV -----------------------------------------------------
    csv_path = out_dir / f"peak_additivity_{experiment}.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v)
                             for k, v in row.items()})

    # --- Console summary -----------------------------------------------------
    print(f"\n=== Peak-based additivity (latency-independent)  "
          f"[{experiment} | {roi_label}] ===")
    print(f"subjects: {n}   site: {roi_label}")
    if cs.skipped:
        print(f"skipped (incomplete condition set): {cs.skipped}")
    for comp in comps:
        r = result[comp]
        print(f"\n  -- {comp} --")
        for tag, d, t, p in (("sum-of-peaks", r["d_sop"], r["t_sop"], r["p_sop"]),
                             ("peak-of-sum ", r["d_pos"], r["t_pos"], r["p_pos"])):
            sem = d.std(ddof=1) / np.sqrt(n)
            print(f"    static  {tag}:  D = {d.mean():+.3f} +/- {sem:.3f} uV   "
                  f"t({n - 1})={t:+.2f}  p={p:.4f} {stars(p)}")
        a, b, c = r["coef_sop"]
        print(f"    learned sum-of-peaks:  a(audio)={a:+.3f}  b(visual)={b:+.3f}  "
              f"c={c:+.3f} uV   R2={r['r2l_sop']:.3f}")
        a, b, c = r["coef_pos"]
        print(f"    learned peak-of-sum :  a(audio)={a:+.3f}  b(visual)={b:+.3f}  "
              f"c={c:+.3f} uV   R2={r['r2l_pos']:.3f}")
        print(f"    latency: VAO vs AO  r={r['r_lat_ao'][0]:+.2f} (p={r['r_lat_ao'][1]:.3f})"
              f"   VAO vs VO  r={r['r_lat_vo'][0]:+.2f} (p={r['r_lat_vo'][1]:.3f})")
        print(f"    mean latency (ms): AO={r['ao_lat'].mean()*1e3:.0f}  "
              f"VO={r['vo_lat'].mean()*1e3:.0f}  VAO={r['vao_lat'].mean()*1e3:.0f}")
    print(f"\nwritten: {csv_path.relative_to(io.PROJECT_ROOT)}")

    # --- Group-level summary CSV --------------------------------------------
    summary_path = out_dir / f"peak_additivity_summary_{experiment}.csv"
    with summary_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["experiment", "component", "model", "prediction",
                         "metric_1", "value_1", "metric_2", "value_2",
                         "metric_3", "value_3"])
        for comp in comps:
            r = result[comp]
            for tag, d, t, p in (("SoP", r["d_sop"], r["t_sop"], r["p_sop"]),
                                 ("PoS", r["d_pos"], r["t_pos"], r["p_pos"])):
                writer.writerow([experiment, comp, "static", tag,
                                 "mean_D_uV", f"{d.mean():.5f}",
                                 "t", f"{t:.4f}", "p", f"{p:.5f}"])
            for tag, coef, r2 in (("SoP", r["coef_sop"], r["r2l_sop"]),
                                  ("PoS", r["coef_pos"], r["r2l_pos"])):
                writer.writerow([experiment, comp, "learned", tag,
                                 "a_audio", f"{coef[0]:.5f}",
                                 "b_visual", f"{coef[1]:.5f}", "R2", f"{r2:.5f}"])
    print(f"written: {summary_path.relative_to(io.PROJECT_ROOT)}")

    _make_figure(experiment, cs, result, out_dir, roi_label, comps)


def _make_figure(experiment: str, cs: io.ConditionSet, result: dict,
                 out_dir, roi_label: str, comps: dict) -> None:
    n = cs.n_subjects
    comp_names = list(comps)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle(f"Peak-based additivity  -  {experiment}  "
                 f"({roi_label}, n={n})")

    # Panel 1: static deviation D for each component x prediction definition.
    ax = axes[0]
    x = np.arange(len(comp_names))
    for off, key, lab, col in ((-0.18, "d_sop", "sum-of-peaks", "tab:blue"),
                               (+0.18, "d_pos", "peak-of-sum", "tab:orange")):
        means = [result[c][key].mean() for c in comp_names]
        sems = [result[c][key].std(ddof=1) / np.sqrt(n) for c in comp_names]
        ax.bar(x + off, means, width=0.34, yerr=sems, capsize=4,
               color=col, alpha=0.8, label=lab)
        for j, c in enumerate(comp_names):
            p = result[c]["p_sop" if key == "d_sop" else "p_pos"]
            ax.annotate(stars(p), (x[j] + off, means[j]),
                        textcoords="offset points",
                        xytext=(0, 6 if means[j] >= 0 else -12),
                        ha="center", fontsize=9)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(comp_names)
    ax.set_ylabel("D = peak(VAO) - predicted  (uV)")
    ax.set_title("Static: deviation from additivity")
    ax.legend(fontsize=8)

    # Panel 2: learned weights a (audio) and b (visual).
    ax = axes[1]
    width = 0.2
    specs = (("coef_sop", 0, "a SoP", "tab:blue", None),
             ("coef_sop", 1, "b SoP", "tab:green", None),
             ("coef_pos", 0, "a PoS", "tab:blue", "//"),
             ("coef_pos", 1, "b PoS", "tab:green", "//"))
    for k, (key, ci, lab, col, hatch) in enumerate(specs):
        vals = [result[c][key][ci] for c in comp_names]
        ax.bar(x + (k - 1.5) * width, vals, width=width, color=col,
               alpha=0.8, hatch=hatch, label=lab)
    ax.axhline(1.0, color="0.5", ls="--", lw=1, label="strict additivity (=1)")
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(comp_names)
    ax.set_ylabel("Fitted weight")
    ax.set_title("Learned weights (fit across subjects)")
    ax.legend(fontsize=7, ncol=2)

    # Panel 3: peak-latency scatter for the earliest component.
    ax = axes[2]
    first = comp_names[0]
    r = result[first]
    ax.scatter(r["ao_lat"] * 1e3, r["vao_lat"] * 1e3, color="tab:blue", s=24,
               label=f"vs AO  r={r['r_lat_ao'][0]:+.2f}")
    ax.scatter(r["vo_lat"] * 1e3, r["vao_lat"] * 1e3, color="tab:green", s=24,
               label=f"vs VO  r={r['r_lat_vo'][0]:+.2f}")
    lo, hi = comps[first]["window"]
    ax.plot([lo * 1e3, hi * 1e3], [lo * 1e3, hi * 1e3], color="0.5", ls="--",
            lw=1, label="identity")
    ax.set_xlabel(f"Unimodal {first} latency (ms)")
    ax.set_ylabel(f"VAO {first} latency (ms)")
    ax.set_title(f"{first} peak latency tracking")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig_path = out_dir / f"peak_additivity_{experiment}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"written: {fig_path.relative_to(io.PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", choices=(*io.EXPERIMENTS, "both"), default="both")
    args = parser.parse_args()
    experiments = io.EXPERIMENTS if args.exp == "both" else (args.exp,)
    for experiment in experiments:
        for site in SITES:
            analyse(experiment, site)


if __name__ == "__main__":
    main()
