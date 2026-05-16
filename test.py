from __future__ import annotations

import configparser
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MPL_CACHE = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

try:
    import mne
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing Python package: {exc.name}\n"
        "Activate the EEG environment first:\n"
        "  conda activate eeg_audivisual\n"
        "  python test.py"
    ) from exc


RAW_BDF = PROJECT_ROOT / "Audvissen_Experiments/Exp1/Raw EEG Files/VP01.bdf"
AVERAGE_VHDR = (
    PROJECT_ROOT
    / "Audvissen_Experiments/Exp1/Preprocessed Files Averages/Vp1_AO.vhdr"
)


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


def _read_vectorized_ascii_data(path: Path, ch_names: list[str], units: list[str]) -> np.ndarray:
    rows = []
    data_ch_names = []

    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            parts = line.split()
            if not parts:
                continue
            data_ch_names.append(parts[0])
            rows.append([float(value) for value in parts[1:]])

    if data_ch_names != ch_names:
        raise ValueError(
            "Channel names in the data file do not match the .vhdr channel order."
        )

    data = np.asarray(rows, dtype=float)
    scales = np.asarray([_unit_scale_to_volts(unit) for unit in units])[:, np.newaxis]
    return data * scales


def _read_markers(
    path: Path, sfreq: float, ch_names: list[str]
) -> tuple[mne.Annotations, list[dict[str, object]]]:
    marker_header = _read_brainvision_ini(path)
    if "Marker Infos" not in marker_header:
        return mne.Annotations([], [], []), []

    onsets = []
    durations = []
    descriptions = []
    annotation_ch_names = []
    marker_rows: list[dict[str, object]] = []

    marker_items = marker_header["Marker Infos"].items()
    sorted_markers = sorted(marker_items, key=lambda item: int(item[0].removeprefix("Mk")))

    for _, value in sorted_markers:
        fields = _split_brainvision_value(value)
        marker_type = fields[0]
        description = fields[1] if len(fields) > 1 else ""
        position = int(fields[2]) if len(fields) > 2 and fields[2] else 1
        size = int(fields[3]) if len(fields) > 3 and fields[3] else 1
        channel = int(fields[4]) if len(fields) > 4 and fields[4] else 0

        label = marker_type if not description else f"{marker_type}/{description}"
        onsets.append((position - 1) / sfreq)
        durations.append(size / sfreq)
        descriptions.append(label)
        annotation_ch_names.append([ch_names[channel - 1]] if channel else [])
        marker_rows.append(
            {
                "label": label,
                "sample": position,
                "time_s": (position - 1) / sfreq,
                "channel": ch_names[channel - 1] if channel else None,
            }
        )

    annotations = mne.Annotations(
        onset=onsets,
        duration=durations,
        description=descriptions,
        ch_names=annotation_ch_names,
    )
    return annotations, marker_rows


def read_brainvision_ascii_average(vhdr_path: Path) -> tuple[mne.EvokedArray, list[dict[str, object]]]:
    """Read the averaged BrainVision ASCII/vectorized exports produced by Analyzer."""
    header = _read_brainvision_ini(vhdr_path)
    common = header["Common Infos"]

    if common.get("DataFormat") != "ASCII" or common.get("DataOrientation") != "VECTORIZED":
        raise ValueError("This helper only supports ASCII BrainVision files in vectorized order.")

    ch_names, units = _channel_info(header)
    sfreq = 1_000_000 / common.getfloat("SamplingInterval")
    data_path = vhdr_path.with_name(common["DataFile"])
    marker_path = vhdr_path.with_name(common["MarkerFile"])

    data = _read_vectorized_ascii_data(data_path, ch_names, units)
    expected_points = common.getint("DataPoints")
    if data.shape != (len(ch_names), expected_points):
        raise ValueError(
            f"Expected {(len(ch_names), expected_points)} data points, got {data.shape}."
        )

    _annotations, markers = _read_markers(marker_path, sfreq, ch_names)
    time_zero_markers = [row for row in markers if row["label"] == "Time 0"]
    tmin = 0.0
    if time_zero_markers:
        tmin = -float(time_zero_markers[0]["time_s"])

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    nave = common.getint("AveragedSegments", fallback=1)
    evoked = mne.EvokedArray(
        data,
        info,
        tmin=tmin,
        nave=nave,
        comment=vhdr_path.stem,
        verbose="ERROR",
    )
    return evoked, markers


def check_torch() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        print("PyTorch: not installed in this interpreter")
        return

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")


def main() -> int:
    print(f"Python: {sys.version.split()[0]} at {sys.executable}")
    print(f"MNE: {mne.__version__}")
    check_torch()

    print("\nRaw BDF smoke test")
    raw = mne.io.read_raw_bdf(RAW_BDF, preload=False, verbose="ERROR")
    duration_s = raw.n_times / raw.info["sfreq"]
    print(f"File: {RAW_BDF.relative_to(PROJECT_ROOT)}")
    print(f"Channels: {len(raw.ch_names)}")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Duration: {duration_s:.1f} s")
    print(f"Annotations: {len(raw.annotations)}")

    print("\nAveraged BrainVision ASCII smoke test")
    evoked, markers = read_brainvision_ascii_average(AVERAGE_VHDR)
    print(f"File: {AVERAGE_VHDR.relative_to(PROJECT_ROOT)}")
    print(f"Channels: {len(evoked.ch_names)}")
    print(f"Sampling rate: {evoked.info['sfreq']} Hz")
    print(f"Samples: {evoked.data.shape[1]}")
    print(f"Time range: {evoked.times[0]:.3f} to {evoked.times[-1]:.3f} s")
    print(f"Averaged trials: {evoked.nave}")
    print(f"Markers: {len(markers)}")
    for marker in markers[:5]:
        channel = "" if marker["channel"] is None else f", channel={marker['channel']}"
        print(f"  {marker['label']} at sample {marker['sample']} ({marker['time_s']:.3f} s{channel})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
