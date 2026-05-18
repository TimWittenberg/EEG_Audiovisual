# EEG Audiovisual Additivity Analyses

This repository contains supplementary EEG analyses for a paper on sensory
attenuation — the suppression of the N1 (and P2) event-related potentials — to
self-initiated sensory events. Across two EEG experiments, the paper compares
sensory attenuation for unisensory auditory and unisensory visual outcomes with
attenuation for multisensory audiovisual outcomes, extending sensory-attenuation
research from single-modality paradigms to the multisensory case. A central
question is whether combining auditory and visual signals increases attenuation,
and which modality drives the audiovisual effect.

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
   VAO = a*AO + b*VO
   ```

2. **Self-initiated, motor-corrected responses**

   ```text
   MVA = a*MA + b*MV
   ```

3. **Attenuation waves**

   ```text
   VAO - MVA = a*(AO - MA) + b*(VO - MV)
   ```

The third level is the most direct test of the paper's attenuation claim,
because it asks whether the audiovisual attenuation effect itself can be
explained as a weighted combination of auditory and visual attenuation.

For each level, the project uses three complementary analysis approaches:

1. **Static additivity tests**
   The strict additive case, with both weights fixed at `a = b = 1`: the target
   audiovisual waveform is compared against the unweighted sum of the auditory
   and visual waveforms. Deviations from this model indicate multisensory
   interaction beyond a simple linear sum.

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

This project asks whether the audiovisual N1 attenuation measured at Cz mostly
reflects the auditory modality. It does — and not as a measurement-site
artefact, since the posterior ROI shows the same pattern. That conclusion rests
on the paper's main result (there is no visual attenuation to begin with) and is
*corroborated* by the additivity analyses below, not carried by them: the
regression fits on the attenuation wave are noisy and weak (see caveat).

**1. The audiovisual attenuation is auditory-carried; the visual modality adds
essentially nothing.**
Reconstructing the audiovisual attenuation as `a*(AO-MA) + b*(VO-MV)`, the
visual-only fit (`a = 0`) explains essentially none of it — visual-only R² ≈ 0
at every site in both experiments. At Cz the audio-only fit retains clearly more
(R² ≈ 0.15–0.25); at the posterior ROI neither modality predicts the attenuation
well. The fitted weights agree, `a > b`, but this reaches significance only at
Cz in Experiment 1 (a = 0.64 vs b = 0.21, p = .003) — it is a trend at Cz in
Experiment 2 (p = .10) and not significant at the ROI. So the weight asymmetry
alone is weak evidence; the robust statement is the model-free one — the
unimodal visual attenuation `VO - MV` is itself ≈ 0, so the sum
`(AO - MA) + (VO - MV)` is necessarily carried by its auditory term. The
per-channel topographic map shows auditory weighting across almost the whole
scalp for the attenuation, but it is descriptive — not channel-wise
significance-tested.

**2. The audiovisual responses themselves are not strictly additive.**
For both externally-initiated (`replay`) and self-initiated (`self`) responses,
the observed audiovisual waveform departs significantly from the summed auditory
and visual waveforms at the auditory site **Cz**. The N1 deviation is reliable in
every raw condition (p ≤ .005); the P2 deviation in all but one (p < .01 — the
exception being self-initiated Experiment 1, p ≈ .08). The departure is largest
around the auditory N1–P2 complex. The
*direction* of the departure is left undescribed on purpose: the
component-window measure mixes amplitude and latency differences, so calling it
sub- or super-additive would over-read it. This non-additivity is
fronto-central — it is not reliable at the posterior ROI.

**3. The attenuation effect shows no _detectable_ departure from additivity.**
For the attenuation level `VAO - MVA = (AO - MA) + (VO - MV)`, no component-level
deviation from the strict sum is reliable at any site in either experiment (all
deviations |D| ≤ 0.9 µV, all p ≥ .24). This is a secondary, weak result: a
*failure to reject* strict additivity is not positive evidence for it. The only
falsifiable additivity claim is the strict one, `a = b = 1`; a freely fitted
`a`, `b` is not a test of additivity at all. The
attenuation waves are noise-dominated (see caveat), the strict test has little
power, and the fitted weights sit well below 1 — so strict additivity is, if
anything, unlikely. This project does not claim it; the reportable point is only
that no multisensory interaction is *detectable* at either site.

**Caveat.** Attenuation waveforms are differences of difference waves and carry
low signal-to-noise. Even with both weights free, the linear fit reaches only
R² ≈ 0.15–0.34 for the attenuation, against R² ≈ 0.6–0.77 for the raw
responses — the model itself is sound, but the attenuation is mostly noise. The
strict-additive R² is strongly negative, and least-squares weights are biased
toward zero by noise in the predictors. "Additive" here means *no detectable
departure* from additivity — a failure to reject, established by the
component-window test — rather than a precise quantitative fit. Experiment 1
(n = 22) and Experiment 2 (n = 30) agree throughout, with Experiment 2 the
cleaner of the two.

## Repository Structure

```text
analysis/
  io_utils.py                 Shared loading/configuration utilities
  static_additivity.py        Fixed AV = A + V tests
  learned_additivity.py       Learned AV = a*A + b*V models
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

The EEG data directory and local paper/literature folder are intentionally
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
python analysis/topographic_additivity.py --exp both --model all
```

Outputs are written to `results/`.

## Data Availability

The EEG data are not included in this repository. They are excluded because of
file size and data-access restrictions. Anyone who wants to recreate the analyses
should contact the author or maintainer of this project to request access to the
underlying EEG data.
