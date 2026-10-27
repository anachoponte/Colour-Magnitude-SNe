# Production run: 2026-10-26 — uncertainty-aware noise injection

**Source population:** `results/dual_59093040` (BlueBEAR), 10 classes, `--n 100000` each,
same scale as the prior dual-schema run (`dual_58818282`). Generated via
`scripts/run_dual_schema_bluebear.sh` at commit `e1970db`.

**What changed since the last dual-schema run (`dual_58818282`, 2026-10-25):**
- `cmsne/multicolour.py::event_multicolour` and `cmsne/trajectory.py::event_trajectory`
  now both emit `*_err` measurement-uncertainty fields for every colour and magnitude
  feature (previously discarded photometric error, now propagated through).
- `ColourClassifier`/`TrajectoryClassifier` train on a Monte Carlo noise-injection
  replication of the population (each event expanded into `n_uncertainty_replicates`
  noisy copies) on both the tree and the local-density (KDE) blend paths, instead of
  treating every measurement as an exact point estimate.
- Spot-checked: 100% of finite colour/magnitude values across all 10 classes also carry
  a finite, positive `*_err` (both the static and trajectory schemas).
- See PR #38 and `docs/superpowers/specs/2026-10-26-uncertainty-aware-noise-injection-design.md`
  for the full design, implementation, and real-data validation writeup (headline finding:
  the bright+red corner's `P(lensed)>0.9` criterion still fails, now understood to be driven
  by importance-sampling weight concentration rather than measurement noise).

**Figures:** full standard set from `scripts/plot_evolving_boundaries.py` against this
population, both `--target lensed_ia` (default, bare filenames) and `--target all_lensed`
(`_alllensed` suffix) — heatmap, sources, corner, and resampled variants for all five
adjacent-band colour pairs (ug, gr, ri, iz, zy), 55 PNGs total.

**For comparison:** the un-dated `docs/figures/boundary_*.png` files at the repo root are
from the prior run (`dual_58818282`, no noise injection) and are left as-is; this folder is
the noise-injection comparison point. Future production runs should add their own dated
`docs/figures/runs/<date>_<short-descriptor>/` folder alongside this one so changes can be
tracked run over run.
