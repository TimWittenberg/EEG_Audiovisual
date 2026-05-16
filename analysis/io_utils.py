"""Shared I/O and configuration for the audiovisual additivity analysis.

This module is strictly READ-ONLY with respect to the source recordings: it
never writes to or modifies anything under ``Audvissen_Experiments/``. All
analysis output is written by the individual scripts into ``results/``.

It parses the averaged BrainVision ASCII exports produced by BrainVision
Analyzer (the files in the "Preprocessed Files Averages" folders, which carry a
``.bdf`` extension but are in fact plain-text ASCII matrices).
"""
from __future__ import annotations

import configparser
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "Audvissen_Experiments"
RESULTS_ROOT = PROJECT_ROOT / "results"
AVERAGES_SUBDIR = "Preprocessed Files Averages"

# Keep matplotlib's cache inside the project (same convention as test.py).
_MPL_CACHE = PROJECT_ROOT / ".cache" / "matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

# --- Analysis configuration --------------------------------------------------
EXPERIMENTS = ("Exp1", "Exp2")

# Replay / externally-initiated conditions. The additive model asks whether the
# audiovisual response equals the sum of its unimodal counterparts:
#     VAO  ?=  AO + VO
AUDIOVISUAL_CONDITION = "VAO"          # combined auditory + visual (replay)
AUDITORY_CONDITION = "AO"              # auditory-only (replay)
VISUAL_CONDITION = "VO"                # visual-only (replay)
REPLAY_CONDITIONS = (AUDITORY_CONDITION, VISUAL_CONDITION, AUDIOVISUAL_CONDITION)

# Posterior / visual ROI: occipital and parieto-occipital sites where the
# visual evoked response is maximal (the manuscript's posterior ROI).
ROI_CHANNELS = ("O1", "O2", "Oz", "PO3", "PO4")

# ERP components by scalp domain. The auditory N1/P2 are measured at the
# fronto-central electrode (Cz); the visual P1/N2 at the posterior ROI. Each
# component carries a polarity (+1 = positive peak, -1 = negative trough) and a
# per-experiment mean-amplitude window in seconds -- all manuscript values
# (placed around the difference-wave / grand-average maxima).
COMPONENTS = {
    "auditory": {
        "N1": dict(polarity=-1, windows={"Exp1": (0.153, 0.193),
                                         "Exp2": (0.166, 0.206)}),
        "P2": dict(polarity=+1, windows={"Exp1": (0.242, 0.282),
                                         "Exp2": (0.248, 0.288)}),
    },
    "visual": {
        "P1": dict(polarity=+1, windows={"Exp1": (0.106, 0.146),
                                         "Exp2": (0.122, 0.162)}),
        "N2": dict(polarity=-1, windows={"Exp1": (0.226, 0.266),
                                         "Exp2": (0.202, 0.242)}),
    },
}


def components_for(domain: str, experiment: str) -> dict[str, dict]:
    """Ordered {name: {'polarity': int, 'window': (lo, hi)}} for a domain."""
    return {
        name: {"polarity": spec["polarity"], "window": spec["windows"][experiment]}
        for name, spec in COMPONENTS[domain].items()
    }


# --- BrainVision ASCII reader ------------------------------------------------
def _read_brainvision_ini(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(
        interpolation=None,
        comment_prefixes=(";",),
        inline_comment_prefixes=(),
        strict=False,
    )
    parser.optionxform = str
    lines = path.read_text(encoding="utf-8-sig").splitlines(keepends=True)
    first_section = next(
        idx for idx, line in enumerate(lines) if line.lstrip().startswith("[")
    )
    parser.read_string("".join(lines[first_section:]), source=str(path))
    return parser


def _split_brainvision_value(value: str) -> list[str]:
    return [part.replace(r"\1", ",").strip() for part in value.split(",")]


def _unit_scale_to_volts(unit: str) -> float:
    normalized = (
        unit.strip()
        .lower()
        .replace("\N{MICRO SIGN}", "u")
        .replace("\N{GREEK SMALL LETTER MU}", "u")
    )
    if normalized in {"uv", "microv"}:
        return 1e-6
    if normalized == "mv":
        return 1e-3
    return 1.0


def _channel_info(header: configparser.ConfigParser) -> tuple[list[str], list[str]]:
    n_channels = header["Common Infos"].getint("NumberOfChannels")
    channel_section = header["Channel Infos"]
    names: list[str] = []
    units: list[str] = []
    for channel_idx in range(1, n_channels + 1):
        fields = _split_brainvision_value(channel_section[f"Ch{channel_idx}"])
        names.append(fields[0])
        units.append(fields[3] if len(fields) > 3 else "")
    return names, units


def _read_vectorized_ascii_data(
    path: Path, ch_names: list[str], units: list[str]
) -> np.ndarray:
    rows: list[list[float]] = []
    data_ch_names: list[str] = []
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            parts = line.split()
            if not parts:
                continue
            data_ch_names.append(parts[0])
            rows.append([float(value) for value in parts[1:]])
    if data_ch_names != ch_names:
        raise ValueError(
            f"{path.name}: channel names in the data file do not match the .vhdr order."
        )
    data = np.asarray(rows, dtype=float)
    scales = np.asarray([_unit_scale_to_volts(unit) for unit in units])[:, np.newaxis]
    return data * scales  # volts


def _read_time_zero(marker_path: Path, sfreq: float) -> float:
    """Return tmin (s): the time of the first sample relative to stimulus onset."""
    if not marker_path.exists():
        return 0.0
    marker_header = _read_brainvision_ini(marker_path)
    if "Marker Infos" not in marker_header:
        return 0.0
    for _, value in marker_header["Marker Infos"].items():
        fields = _split_brainvision_value(value)
        if fields and fields[0] == "Time 0":
            position = int(fields[2]) if len(fields) > 2 and fields[2] else 1
            return -(position - 1) / sfreq
    return 0.0


def read_average(vhdr_path: Path) -> tuple[list[str], float, np.ndarray, np.ndarray]:
    """Read one averaged BrainVision ASCII export.

    Returns (channel_names, sfreq, times_seconds, data_volts) where data has
    shape (n_channels, n_times).
    """
    header = _read_brainvision_ini(vhdr_path)
    common = header["Common Infos"]
    if common.get("DataFormat") != "ASCII" or common.get("DataOrientation") != "VECTORIZED":
        raise ValueError(f"{vhdr_path.name}: expected ASCII/VECTORIZED BrainVision data.")

    ch_names, units = _channel_info(header)
    sfreq = 1_000_000 / common.getfloat("SamplingInterval")
    n_points = common.getint("DataPoints")

    data = _read_vectorized_ascii_data(vhdr_path.with_name(common["DataFile"]), ch_names, units)
    if data.shape != (len(ch_names), n_points):
        raise ValueError(
            f"{vhdr_path.name}: expected {(len(ch_names), n_points)} samples, got {data.shape}."
        )

    tmin = _read_time_zero(vhdr_path.with_name(common["MarkerFile"]), sfreq)
    times = tmin + np.arange(n_points) / sfreq
    return ch_names, sfreq, times, data


# --- Multi-subject loading ---------------------------------------------------
@dataclass
class ConditionSet:
    """All subjects of one experiment, for a fixed set of conditions.

    ``data`` has shape (n_subjects, n_conditions, n_channels, n_times) in volts.
    """

    experiment: str
    subjects: list[str]
    conditions: tuple[str, ...]
    ch_names: list[str]
    sfreq: float
    times: np.ndarray
    data: np.ndarray
    skipped: list[str]

    def cond(self, name: str) -> np.ndarray:
        """Return data for one condition: shape (n_subjects, n_channels, n_times)."""
        return self.data[:, self.conditions.index(name)]

    def roi_indices(self, channels: tuple[str, ...] = ROI_CHANNELS) -> list[int]:
        return [self.ch_names.index(c) for c in channels if c in self.ch_names]

    @property
    def n_subjects(self) -> int:
        return len(self.subjects)


def _subject_sort_key(subject: str) -> int:
    match = re.search(r"(\d+)", subject)
    return int(match.group(1)) if match else 0


def load_condition_set(
    experiment: str, conditions: tuple[str, ...] = REPLAY_CONDITIONS
) -> ConditionSet:
    """Load every subject of ``experiment`` that has all requested conditions."""
    avg_dir = DATA_ROOT / experiment / AVERAGES_SUBDIR
    if not avg_dir.is_dir():
        raise FileNotFoundError(f"Averages folder not found: {avg_dir}")

    by_subject: dict[str, dict[str, Path]] = {}
    for vhdr in sorted(avg_dir.glob("*.vhdr")):
        subject, _, condition = vhdr.stem.rpartition("_")
        if subject and condition in conditions:
            by_subject.setdefault(subject, {})[condition] = vhdr

    complete = sorted(
        (s for s, c in by_subject.items() if all(k in c for k in conditions)),
        key=_subject_sort_key,
    )
    if not complete:
        raise RuntimeError(f"No subject in {avg_dir} has all of {conditions}.")

    ref_ch: list[str] | None = None
    ref_times: np.ndarray | None = None
    ref_sfreq: float | None = None
    used: list[str] = []
    skipped: list[str] = []
    stacked: list[np.ndarray] = []

    for subject in complete:
        per_condition: list[np.ndarray] = []
        ok = True
        for condition in conditions:
            ch_names, sfreq, times, data = read_average(by_subject[subject][condition])
            if ref_ch is None:
                ref_ch, ref_sfreq, ref_times = ch_names, sfreq, times
            if ch_names != ref_ch or data.shape[1] != ref_times.size:
                ok = False
                break
            if not np.allclose(times, ref_times, atol=1e-6):
                # Same length but a different time axis -> not safely comparable.
                ok = False
                break
            per_condition.append(data)
        if ok:
            stacked.append(np.stack(per_condition))  # (n_cond, n_ch, n_time)
            used.append(subject)
        else:
            skipped.append(subject)

    assert ref_ch is not None and ref_times is not None and ref_sfreq is not None
    return ConditionSet(
        experiment=experiment,
        subjects=used,
        conditions=tuple(conditions),
        ch_names=ref_ch,
        sfreq=ref_sfreq,
        times=ref_times,
        data=np.stack(stacked),
        skipped=skipped,
    )


# --- Small helpers -----------------------------------------------------------
def window_mean(data: np.ndarray, times: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    """Mean amplitude over a time window; averages the last (time) axis."""
    lo, hi = window
    mask = (times >= lo) & (times <= hi)
    if not mask.any():
        raise ValueError(f"No samples fall inside window {window}.")
    return data[..., mask].mean(axis=-1)


def r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Coefficient of determination of a prediction (can be negative)."""
    y = y.ravel()
    y_hat = y_hat.ravel()
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


if __name__ == "__main__":
    # Quick sanity check: load both experiments and print what was found.
    for exp in EXPERIMENTS:
        cs = load_condition_set(exp)
        print(f"{exp}: {cs.n_subjects} subjects x {len(cs.conditions)} conditions")
        print(f"  conditions : {cs.conditions}")
        print(f"  channels   : {len(cs.ch_names)}  sfreq={cs.sfreq:g} Hz")
        print(f"  epoch      : {cs.times[0]*1e3:.0f}..{cs.times[-1]*1e3:.0f} ms"
              f" ({cs.times.size} samples)")
        print(f"  data shape : {cs.data.shape}")
        if cs.skipped:
            print(f"  skipped    : {cs.skipped}")
        print()
