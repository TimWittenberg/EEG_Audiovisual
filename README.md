# EEG Audiovisual Additivity Analyses

This repository contains supplementary EEG analyses for a manuscript on N1
suppression to self-generated audiovisual sensory events. The manuscript examines
whether sensory attenuation for audiovisual outcomes is comparable to attenuation
for unisensory auditory and visual outcomes.

The analyses in this project were developed to address a reviewer concern about
the interpretation of the audiovisual N1 effect. In the manuscript, the N1 was
mainly quantified at Cz, a fronto-central electrode where auditory N1 responses
are typically prominent. The reviewer therefore asked whether the audiovisual N1
effect might mostly reflect the auditory component, while visual contributions may
be better captured at posterior/occipital sites.

## What This Project Shows

The project tests this question with three complementary analysis approaches:

1. **Static additivity tests**
   The audiovisual replay response is compared against the summed unisensory
   auditory and visual replay responses:

   ```text
   AV ?= A + V
   ```

   Deviations from this model indicate multisensory interaction beyond a simple
   linear sum of auditory and visual responses.

2. **Learned additive models**
   The audiovisual response is reconstructed as a weighted combination of the
   auditory and visual responses:

   ```text
   AV = a * A + b * V
   ```

   The fitted weights `a` and `b` show whether the audiovisual response is more
   strongly explained by the auditory or visual modality at a given scalp region.

3. **Topographic additivity**
   The same learned model is fitted separately for each EEG channel. This shows
   how auditory and visual weighting changes across the scalp.

Together, the analyses support a spatially specific interpretation:

- At **Cz**, especially in the N1 time range, the audiovisual response is mainly
  auditory-weighted.
- At **posterior visual sites** (`O1`, `O2`, `Oz`, `PO3`, `PO4`), the pattern shifts
  toward stronger visual weighting.
- This supports the manuscript's conclusion that the Cz audiovisual N1 attenuation
  is strongly influenced by the auditory modality, while also acknowledging that
  visual contributions are better captured over posterior scalp regions.

## Repository Structure

```text
analysis/
  io_utils.py                 Shared loading/configuration utilities
  static_additivity.py        Fixed AV = A + V tests
  learned_additivity.py       Learned AV = a*A + b*V models
  peak_additivity.py          Peak-based additivity tests
  topographic_additivity.py   Per-channel topographic weighting analysis

results/
  Cz/                         Cz-focused auditory/N1-P2 outputs
  ROI/                        Posterior visual ROI outputs
  topographic_additivity_*.csv/.png

related_work/
  Manuscript and supporting literature
```

The EEG data directory is intentionally excluded from version control.

## Recreating the Analyses

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate eeg_audivisual
```

Run the analyses:

```bash
python analysis/static_additivity.py --exp both
python analysis/learned_additivity.py --exp both
python analysis/peak_additivity.py --exp both
python analysis/topographic_additivity.py --exp both
```

Outputs are written to `results/`.

## Data Availability

The EEG data are not included in this repository. They are excluded because of
file size and data-access restrictions. Anyone who wants to recreate the analyses
should contact the author or maintainer of this project to request access to the
underlying EEG data.
