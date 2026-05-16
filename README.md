# EEG Audiovisual Additivity Analyses

This repository contains supplementary EEG analyses for a paper on N1
suppression to self-generated audiovisual sensory events. The paper examines
whether sensory attenuation for audiovisual outcomes is comparable to attenuation
for unisensory auditory and visual outcomes.

The analyses in this project were developed to address a reviewer concern about
the interpretation of the audiovisual N1 effect. In the paper, the N1 was
mainly quantified at Cz, a fronto-central electrode where auditory N1 responses
are typically prominent. The reviewer therefore asked whether the audiovisual N1
effect might mostly reflect the auditory component, while visual contributions may
be better captured at posterior/occipital sites.

## What This Project Shows

The project now runs each additivity analysis at three levels:

1. **Externally initiated / replay responses**

   ```text
   VAO ?= AO + VO
   ```

2. **Self-initiated, motor-corrected responses**

   ```text
   MVA ?= MA + MV
   ```

3. **Attenuation waves**

   ```text
   VAO - MVA ?= (AO - MA) + (VO - MV)
   ```

The third level is the most direct test of the manuscript's attenuation claim,
because it asks whether the audiovisual attenuation effect itself can be
explained as a weighted combination of auditory and visual attenuation.

For each level, the project uses three complementary analysis approaches:

1. **Static additivity tests**
   The target audiovisual waveform is compared against the summed auditory and
   visual waveforms. Deviations from this model indicate multisensory interaction
   beyond a simple linear sum.

2. **Learned additive models**
   The target audiovisual waveform is reconstructed as a weighted combination of
   auditory and visual waveforms:

   ```text
   target = a * audio + b * visual
   ```

   The fitted weights `a` and `b` show whether the audiovisual response is more
   strongly explained by the auditory or visual modality at a given scalp region.

3. **Topographic additivity**
   The same learned model is fitted separately for each EEG channel. This shows
   how auditory and visual weighting changes across the scalp.

## Findings

The analyses separate two questions that turn out to have different answers.

**1. The audiovisual responses themselves are not strictly additive.**
For both externally-initiated (`replay`) and self-initiated (`self`) responses,
the observed audiovisual waveform departs significantly from the summed auditory
and visual waveforms at the auditory site **Cz**. The strict-sum prediction
overshoots the N1 by roughly 2–3 µV and undershoots the P2 by roughly 1.4–2.2 µV
(p < .01 in both experiments). Fitted weights `a` and `b` lie below the additive
value of 1 almost everywhere. The audiovisual sensory response is a *compressed*
version of the linear sum, sharpest around the auditory N1–P2 complex.

**2. The attenuation effect, by contrast, is additive.**
For the attenuation level `VAO - MVA ?= (AO - MA) + (VO - MV)`, no
component-level deviation is reliable at any site in either experiment (all
deviations |D| ≤ 0.9 µV, all p ≥ .24). The non-additivity seen in `replay` and
`self` is shared between the two conditions and cancels in their difference. The
audiovisual attenuation can therefore be treated as the sum of the auditory and
visual attenuations — even though the underlying responses cannot.

**3. The attenuation is audio-dominated across the scalp.**
The fitted audio weight exceeds the visual weight everywhere for the attenuation
level (significantly at Cz in Experiment 1: a = 0.65 vs b = 0.17, p = .002). For
the raw `replay` and `self` responses, audio dominance is fronto-central but
flips to visual dominance over occipital channels; for the attenuation wave this
flip never occurs — audio dominates even at the posterior ROI. The audiovisual
attenuation is carried mainly by the auditory modality.

Together this supports the manuscript's interpretation: the Cz audiovisual N1
attenuation is dominated by the auditory modality, while visual contributions
are better captured over posterior scalp. Crucially, the additivity of the
attenuation effect itself holds at both sites.

**Caveat.** Attenuation waveforms are differences of difference waves and carry
low signal-to-noise: the strict-additive whole-epoch R² is strongly negative,
and least-squares weights are biased downward by noise in the predictors.
"Additive" here means *no detectable departure* from additivity — established by
the noise-robust component-window test — rather than a precise quantitative fit.
Experiment 1 (n = 22) and Experiment 2 (n = 30) agree throughout, with
Experiment 2 the cleaner of the two.

## Repository Structure

```text
analysis/
  io_utils.py                 Shared loading/configuration utilities
  static_additivity.py        Fixed AV = A + V tests
  learned_additivity.py       Learned AV = a*A + b*V models
  peak_additivity.py          Peak-based additivity tests
  topographic_additivity.py   Per-channel topographic weighting analysis

results/
  Cz/
    replay/                   Cz replay-response outputs
    self/                     Cz self-initiated response outputs
    attenuation/              Cz attenuation-wave outputs
  ROI/
    replay/                   Posterior ROI replay-response outputs
    self/                     Posterior ROI self-initiated response outputs
    attenuation/              Posterior ROI attenuation-wave outputs
  topographic/
    replay/                   Per-channel replay-response maps
    self/                     Per-channel self-initiated maps
    attenuation/              Per-channel attenuation-wave maps
```

The EEG data directory and local manuscript/literature folder are intentionally
excluded from version control.

## Recreating the Analyses

Create the conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate eeg_audivisual
```

The environment file specifies the required Python version and analysis
dependencies, including `numpy`, `scipy`, `pandas`, `matplotlib`, `mne`,
`scikit-learn`, `jupyterlab`, and `torch`.

Run the analyses:

```bash
python analysis/static_additivity.py --exp both --model all
python analysis/learned_additivity.py --exp both --model all
python analysis/peak_additivity.py --exp both --model all
python analysis/topographic_additivity.py --exp both --model all
```

Outputs are written to `results/`.

## Data Availability

The EEG data are not included in this repository. They are excluded because of
file size and data-access restrictions. Anyone who wants to recreate the analyses
should contact the author or maintainer of this project to request access to the
underlying EEG data.
