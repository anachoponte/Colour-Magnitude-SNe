# Production run: 2026-10-27 — band-matched CMD features

**Source population:** `results/dual_59253457` (BlueBEAR), 10 classes, `--n 100000` each,
same scale as prior production runs (`dual_59093040`, `dual_58818282`). Generated via
`scripts/run_dual_schema_bluebear.sh` (SLURM array job `59253457_[0-9]`) and plotted via
`scripts/run_plot_boundaries_bluebear.sh` (SLURM job `59253458`) at commit `d54fbe0`.

**What changed since the prior run (`dual_59093040`, 2026-10-26):**
- **Band-Matched CMD Features (Remedy 3):** Rather than evaluating an all-band global
  `peakmag` against colour, each adjacent-band colour $(c_1 - c_2)$ is paired with its
  companion redder band $c_2$ cumulative best (`peakmag_{c2}`), conforming to astronomical
  colour-magnitude diagram conventions (Arendse et al. 2025, Quimby et al. 2014, Mane et al. 2024/2025):
  - $g-r$ paired with $r$ peakmag (`peakmag_r`)
  - $r-i$ paired with $i$ peakmag (`peakmag_i`)
  - $i-z$ paired with $z$ peakmag (`peakmag_z`)
  - $z-y$ paired with $y$ peakmag (`peakmag_y`)
  - $u-g$ paired with $g$ peakmag (`peakmag_g`)
- Per-band snapshot (`{b}{suffix}`, `{b}{suffix}_err`) and cumulative-best
  (`peakmag_{b}{suffix}`, `peakmag_{b}{suffix}_err`) features are recorded at each grid step
  in `cmsne.trajectory::event_trajectory` and propagated into `DualSchemaGenerator`.
- `ColourClassifier` and `TrajectoryClassifier` now accept band-matched magnitude features
  and enforce non-increasing monotonicity constraints across each per-band cumulative best sequence.
- `scripts/plot_evolving_boundaries.py` supports `--mag-mode matched` (default) and displays
  the companion band magnitude label on the x-axis (e.g. `r peakmag (cumulative-best)`).

**Figures:** Full standard set of 55 PNGs from `scripts/plot_evolving_boundaries.py` across
all five adjacent-band colour pairs (`ug`, `gr`, `ri`, `iz`, `zy`), for both `--target lensed_ia`
(default, bare filenames) and `--target all_lensed` (`_alllensed` suffix) — heatmaps, sources,
corner plots, and weighted resamples.
