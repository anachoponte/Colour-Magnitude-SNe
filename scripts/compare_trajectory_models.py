"""Investigate improvements and replacements for the full-trajectory GBDT classifier.

Benchmarks 6 full-trajectory architectures on the 100k/class production dataset
(70/30 held-out test split, rate-weighted evaluation, strictly unlensed_* background):

  1. Baseline GBDT (Current):
     5 adjacent colours x 5 steps + 5 global peakmags (30 features), Isotonic calibration.
     NOTE: unlike the actual deployed cmsne.trajectory.TrajectoryClassifier, this fits a
     bare HistGradientBoostingClassifier directly and does NOT apply the production
     default's uncertainty-aware noise-injection replication (n_uncertainty_replicates=8,
     see cmsne/classifier.py). Its zero-FP numbers are therefore NOT directly comparable
     to the deployed classifier's own operating point (see investigate_purity_tradeoff.py
     / docs/session_writeup.md Finding 2, which uses the real TrajectoryClassifier and
     gets a higher zero-FP yield than this row) -- it isolates the calibration/feature/
     model-architecture choices below from noise injection, which is evaluated separately.
  2. Smooth-Calibrated & Regularised GBDT (Platt / Sigmoid + Leaf Smoothing):
     Same 30 features, replacing jagged Isotonic step-functions with smooth Sigmoid
     (Platt) calibration and larger min_samples_leaf + L2 regularisation.
  3. Physics-Enriched Trajectory GBDT (+ Sigmoid Calibration):
     Adds per-band snapshot magnitudes (u,g,r,i,z,y at each step), explicit fade rates
     (delta_mag = mag(t) - peakmag_band(t)), broad baseline colours, colour evolution
     rates, and photometric errors, totalling 132 features (102 beyond the 30 baseline),
     with smooth Sigmoid calibration.
  4. Spline Logistic GAM (Smooth Generalized Additive Model + Interactions):
     Missing-ness indicator + weighted median imputation + B-Spline basis expansion
     with Ridge-penalised Logistic Regression (smooth, infinitely differentiable log-odds,
     no rectangular axis-aligned box splits).
  5. Mask-Aware Smooth Neural Network (MLP with Missingness Mask + Weighted Loss):
     Concatenates standardized features with binary observation masks [X_std, M] (so missing
     cadence visits are learned natively without tree splits), smooth SiLU/ReLU-like tanh/logistic
     or MLP ensemble trained via weighted log-loss (using scipy/numpy or sklearn sample_weight
     resampling + smooth architecture).
  6. Hybrid Smooth Ensemble (Physics-Enriched GBDT + Smooth Neural MLP / Spline GAM):
     Combines the missing-data tree interactions with the smooth extrapolate-safe
     parametric/neural decision surface.

Evaluates:
  - Full-trajectory ROC AUC
  - Time-evolving ROC AUC at each checkpoint (+0d, +3d, +14d, +30d, +60d)
  - TPR @ 10% FPR, 1% FPR, 0.1% FPR, 0.01% FPR
  - Zero-FPR Sweet Spot recovery (TPR where empirical test FPR == 0, i.e. 100% pure sample!)
  - Annual pure lensed SNe yield in LSST

Outputs JSON summary and comparison figures.
"""
import argparse, json, os, pickle, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.metrics import roc_curve as sklearn_roc_curve

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from cmsne.trajectory import (
    TRAJECTORY_GRID, TRAJECTORY_COLOUR_KEYS, TRAJECTORY_MAGNITUDE_KEYS,
    _step_suffix, reveal_up_to
)
from cmsne.lsst import bands as LSST_BANDS
from cmsne.multicolour import ADJ_COLOURS

CLASSES = ['lensed_Ia', 'unlensed_Ia', 'lensed_sn1bc', 'lensed_sn2l', 'lensed_sn2n',
           'lensed_sn2p', 'unlensed_sn1bc', 'unlensed_sn2l', 'unlensed_sn2n', 'unlensed_sn2p']
UNLENSED_CLASSES = [c for c in CLASSES if c.startswith('unlensed_')]
LENSED_CLASSES = [c for c in CLASSES if c.startswith('lensed_')]

# See scripts/investigate_purity_tradeoff.py's ANNUAL_RATES for the full
# citation/derivation note (same figures, kept in sync between both scripts).
# lensed_ia: ~10/yr (Oguri 2010; Wojtak 2019; Goldstein 2019), ~100,000/yr
# unlensed detected SNe -- both already cited in docs/methods_and_results.md.
# all_lensed: order-of-magnitude placeholder pending issue #43.
ANNUAL_RATES = {
    'all_lensed': {'N_sig': 45.0, 'N_bg': 100_000.0},
    'lensed_ia':  {'N_sig': 10.0, 'N_bg': 100_000.0},
}


def split_events(events, rng, frac_train=0.7):
    idx = rng.permutation(len(events))
    n_train = int(len(events) * frac_train)
    return [events[i] for i in idx[:n_train]], [events[i] for i in idx[n_train:]]


def compute_auc(fpr, tpr):
    fpr, tpr = np.asarray(fpr, float), np.asarray(tpr, float)
    order = np.argsort(fpr)
    fpr, tpr = fpr[order], tpr[order]
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def extract_baseline_features(events):
    """Original 30 trajectory features: 25 adjacent colours + 5 global cumulative peakmags."""
    cols = list(TRAJECTORY_COLOUR_KEYS) + list(TRAJECTORY_MAGNITUDE_KEYS)
    X = np.array([[float(e.get(c, np.nan)) if e.get(c, None) is not None else np.nan
                   for c in cols] for e in events], dtype=float)
    return X, cols


def extract_enriched_features(events, grid=TRAJECTORY_GRID):
    """Physics-enriched trajectory features (causal at every step when passed reveal_up_to):
      1. Baseline 25 adjacent colours + 5 global cumulative peakmags (30)
      2. Per-band snapshot magnitudes u,g,r,i,z,y at each step (30)
      3. Per-band fade from band peak: mag_b(t) - peakmag_b(t) (captures time-dilated decline!) (30)
      4. Non-adjacent broad baseline colours: g-i, r-z, g-z at each step (15)
      5. Colour evolution since earliest available observation in gr, ri, iz (15)
      6. Photometric SNR / uncertainty proxy: r_err, i_err, z_err at each step (15)
    """
    n = len(events)
    base_X, base_cols = extract_baseline_features(events)

    extra_arrays = [base_X]
    extra_names = list(base_cols)

    b_names = [b[-1] for b in LSST_BANDS]  # u, g, r, i, z, y

    # Per-step snapshot mags, fade rates, broad colours, and errors
    snap_dict = {}
    peak_dict = {}
    for day in grid:
        suf = _step_suffix(day)
        for b in b_names:
            k_snap = f'{b}{suf}'
            k_peak = f'peakmag_{b}{suf}'
            arr_snap = np.array([float(e.get(k_snap, np.nan)) for e in events], dtype=float)
            arr_peak = np.array([float(e.get(k_peak, np.nan)) for e in events], dtype=float)
            snap_dict[(b, day)] = arr_snap
            peak_dict[(b, day)] = arr_peak

            extra_arrays.append(arr_snap[:, None])
            extra_names.append(k_snap)

            # Fade from cumulative band peak (>= 0 when fading; 0 at peak/rising)
            fade = arr_snap - arr_peak
            extra_arrays.append(fade[:, None])
            extra_names.append(f'fade_{b}{suf}')

        # Broad wavelength-baseline colours: g-i, r-z, g-z
        for b1, b2 in (('g', 'i'), ('r', 'z'), ('g', 'z')):
            broad_c = snap_dict[(b1, day)] - snap_dict[(b2, day)]
            extra_arrays.append(broad_c[:, None])
            extra_names.append(f'{b1}{b2}_broad{suf}')

        # Measurement uncertainty features for key bands (r, i, z)
        for b in ('r', 'i', 'z'):
            k_err = f'{b}{suf}_err'
            arr_err = np.array([float(e.get(k_err, np.nan)) for e in events], dtype=float)
            extra_arrays.append(arr_err[:, None])
            extra_names.append(k_err)

    # Colour evolution relative to trigger (+0d): c(t) - c(0) for gr, ri, iz
    for pair in ('gr', 'ri', 'iz'):
        c0 = np.array([float(e.get(f'{pair}_d0', np.nan)) for e in events], dtype=float)
        for day in grid[1:]:
            suf = _step_suffix(day)
            ct = np.array([float(e.get(f'{pair}{suf}', np.nan)) for e in events], dtype=float)
            dc = ct - c0
            extra_arrays.append(dc[:, None])
            extra_names.append(f'd_{pair}{suf}')

    X_full = np.hstack(extra_arrays)
    return X_full, extra_names


class MaskImputerScaler:
    """Transforms NaN-bearing tabular trajectory matrix X into [X_standardized_imputed, Missing_Mask].
    Computes rate-weighted mean and std per column on training data so high-weight physical bulk
    drives the standardization rather than zero-weight Monte Carlo outliers.
    """
    def fit(self, X, sample_weight):
        self.n_features_ = X.shape[1]
        self.means_ = np.zeros(self.n_features_)
        self.stds_ = np.ones(self.n_features_)
        w = np.asarray(sample_weight, dtype=float)
        for j in range(self.n_features_):
            col = X[:, j]
            fin = np.isfinite(col) & (w > 0)
            if np.any(fin):
                wf = w[fin]
                cf = col[fin]
                m = np.sum(wf * cf) / np.sum(wf)
                v = np.sum(wf * (cf - m) ** 2) / np.sum(wf)
                self.means_[j] = m
                self.stds_[j] = max(np.sqrt(v), 1e-3)
        return self

    def transform(self, X, include_mask=True):
        mask = np.isfinite(X).astype(np.float32)
        X_imp = np.where(np.isfinite(X), X, self.means_[None, :])
        X_std = np.clip((X_imp - self.means_[None, :]) / self.stds_[None, :], -8.0, 8.0)
        if include_mask:
            return np.hstack([X_std, mask])
        return X_std


class SplineGAMClassifier:
    """Smooth Generalized Additive Model (B-Splines + Missingness Masks + L2 Logistic Regression).
    Produces completely smooth, differentiable probability surfaces with zero axis-aligned box artifacts.
    """
    def __init__(self, n_knots=5, degree=3, C=1.0):
        self.n_knots = n_knots
        self.degree = degree
        self.C = C

    def fit(self, X, y, sample_weight):
        self.prep_ = MaskImputerScaler().fit(X, sample_weight)
        X_std = self.prep_.transform(X, include_mask=False)
        mask = np.isfinite(X).astype(float)

        # Fit B-splines on weighted-resampled representative subset so knots follow physical density
        rng = np.random.default_rng(0)
        p = sample_weight / sample_weight.sum()
        idx_knot = rng.choice(len(X_std), size=min(10000, len(X_std)), replace=True, p=p)
        self.spline_ = SplineTransformer(n_knots=self.n_knots, degree=self.degree, include_bias=False)
        self.spline_.fit(X_std[idx_knot])

        X_sp = self.spline_.transform(X_std)
        # Zero out spline basis functions wherever the original feature was missing (NaN),
        # and append explicit missingness indicators!
        n_basis_per_feat = X_sp.shape[1] // X_std.shape[1]
        mask_expanded = np.repeat(mask, n_basis_per_feat, axis=1)
        X_design = np.hstack([X_sp * mask_expanded, mask])

        self.clf_ = LogisticRegression(C=self.C, max_iter=400, solver='lbfgs')
        self.clf_.fit(X_design, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        X_std = self.prep_.transform(X, include_mask=False)
        mask = np.isfinite(X).astype(float)
        X_sp = self.spline_.transform(X_std)
        n_basis_per_feat = X_sp.shape[1] // X_std.shape[1]
        mask_expanded = np.repeat(mask, n_basis_per_feat, axis=1)
        X_design = np.hstack([X_sp * mask_expanded, mask])
        return self.clf_.predict_proba(X_design)[:, 1]


class WeightedMaskMLPClassifier:
    """Smooth Mask-Aware Neural Network Ensemble (3 diverse MLPs trained on rate-weighted
    importance resamples with [X_standardized, Observation_Mask] inputs).
    Because sklearn 1.0 MLPClassifier does not accept sample_weight directly in fit(),
    we draw 3 large rate-weighted bootstrap training sets (60,000 events each, exact 50/50
    class balance) and average their smooth sigmoid probabilities!
    """
    def __init__(self, n_estimators=3, n_samples=60000, hidden_layer_sizes=(128, 64, 32), alpha=1e-2):
        self.n_estimators = n_estimators
        self.n_samples = n_samples
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha

    def fit(self, X, y, sample_weight):
        self.prep_ = MaskImputerScaler().fit(X, sample_weight)
        X_trans = self.prep_.transform(X, include_mask=True)

        sig_idx = np.where((y == 1) & (sample_weight > 0))[0]
        bg_idx = np.where((y == 0) & (sample_weight > 0))[0]
        p_sig = sample_weight[sig_idx] / sample_weight[sig_idx].sum()
        p_bg = sample_weight[bg_idx] / sample_weight[bg_idx].sum()

        self.models_ = []
        n_half = self.n_samples // 2
        for m in range(self.n_estimators):
            rng = np.random.default_rng(100 + m)
            i_s = rng.choice(sig_idx, size=n_half, replace=True, p=p_sig)
            i_b = rng.choice(bg_idx, size=n_half, replace=True, p=p_bg)
            idx = np.r_[i_s, i_b]
            perm = rng.permutation(len(idx))
            idx = idx[perm]

            mlp = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation='relu',
                alpha=self.alpha,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=60,
                early_stopping=True,
                n_iter_no_change=8,
                random_state=100 + m
            )
            mlp.fit(X_trans[idx], y[idx])
            self.models_.append(mlp)
        return self

    def predict_proba(self, X):
        X_trans = self.prep_.transform(X, include_mask=True)
        preds = [m.predict_proba(X_trans)[:, 1] for m in self.models_]
        return np.mean(preds, axis=0)


def evaluate_scores(y_test, w_test, scores, N_sig_ann, N_bg_ann):
    fpr, tpr, thresholds = sklearn_roc_curve(y_test, scores, sample_weight=w_test)
    roc_auc = compute_auc(fpr, tpr)

    order = np.argsort(fpr)
    fpr_s, tpr_s = fpr[order], tpr[order]

    tpr_10 = float(np.interp(0.10, fpr_s, tpr_s))
    tpr_01 = float(np.interp(0.01, fpr_s, tpr_s))
    tpr_001 = float(np.interp(0.001, fpr_s, tpr_s))
    tpr_0001 = float(np.interp(0.0001, fpr_s, tpr_s))

    # Zero-FPR sweet spot (strictly above max positive-weight background score)
    bg_mask = (y_test == 0) & (w_test > 0)
    sig_mask = (y_test == 1) & (w_test > 0)
    max_bg = float(np.max(scores[bg_mask]))
    pure_sig = sig_mask & (scores > max_bg)
    pure_tpr = float(np.sum(w_test[pure_sig]) / np.sum(w_test[sig_mask]))
    pure_tp_yr = pure_tpr * N_sig_ann

    return {
        'auc': roc_auc,
        'tpr_10pct': tpr_10,
        'tpr_1pct': tpr_01,
        'tpr_01pct': tpr_001,
        'tpr_001pct': tpr_0001,
        'zero_fp_max_bg_score': max_bg,
        'zero_fp_tpr': pure_tpr,
        'zero_fp_tp_per_yr': pure_tp_yr,
        'zero_fp_raw_count': int(np.sum(pure_sig)),
        'fpr_curve': fpr_s,
        'tpr_curve': tpr_s,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', default='docs/figures/model_comparison')
    ap.add_argument('--target', choices=['all_lensed', 'lensed_ia'], default='all_lensed')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.target == 'all_lensed':
        signal_classes = LENSED_CLASSES
        signal_label = 'All Lensed SNe'
        suffix = '_alllensed'
    else:
        signal_classes = ['lensed_Ia']
        signal_label = 'Lensed Type Ia SNe'
        suffix = '_lensed_ia'

    rates = ANNUAL_RATES[args.target]
    N_sig_ann, N_bg_ann = rates['N_sig'], rates['N_bg']

    print(f'Loading production dataset from {args.data_dir} ({signal_label})...')
    pop = {}
    for c in CLASSES:
        with open(os.path.join(args.data_dir, f'{c}.pkl'), 'rb') as f:
            pop[c] = pickle.load(f)['events']

    signal = [e for c in signal_classes for e in pop[c]]
    background = [e for c in UNLENSED_CLASSES for e in pop[c]]

    rng = np.random.default_rng(args.seed)
    tr_sig, te_sig = split_events(signal, rng)
    tr_bg, te_bg = split_events(background, rng)

    train_events = tr_sig + tr_bg
    test_events = te_sig + te_bg

    ws_tr = np.array([e['weight'] for e in tr_sig], dtype=float)
    wb_tr = np.array([e['weight'] for e in tr_bg], dtype=float)
    ws_tr_bal = ws_tr * (wb_tr.sum() / ws_tr.sum())

    y_train = np.r_[np.ones(len(tr_sig)), np.zeros(len(tr_bg))]
    w_train = np.r_[ws_tr_bal, wb_tr]

    ws_te = np.array([e['weight'] for e in te_sig], dtype=float)
    wb_te = np.array([e['weight'] for e in te_bg], dtype=float)
    y_test = np.r_[np.ones(len(te_sig)), np.zeros(len(te_bg))]
    w_test = np.r_[ws_te, wb_te]  # Note: roc_curve normalizes within class automatically

    print('Extracting baseline (30-dim) and physics-enriched (132-dim) feature matrices...')
    X_tr_base, base_cols = extract_baseline_features(train_events)
    X_te_base, _ = extract_baseline_features(test_events)

    X_tr_enr, enr_cols = extract_enriched_features(train_events)
    X_te_enr, _ = extract_enriched_features(test_events)
    print(f'Baseline features: {X_tr_base.shape[1]}, Physics-Enriched features: {X_tr_enr.shape[1]}')

    # Also pre-extract step-by-step partial test matrices for temporal evolution evaluation!
    X_te_base_steps = {}
    X_te_enr_steps = {}
    for k, day in enumerate(TRAJECTORY_GRID):
        partial_te = [reveal_up_to(e, k, TRAJECTORY_GRID) for e in test_events]
        X_te_base_steps[day], _ = extract_baseline_features(partial_te)
        X_te_enr_steps[day], _ = extract_enriched_features(partial_te)

    # To train models that are ALSO evaluated at partial epochs (+0d, +3d, +14d, ...),
    # let's check both standard full-trajectory training AND multi-step Masked-Prefix Augmentation!
    # Wait: in current TrajectoryClassifier, it is trained ONLY on fully-revealed trajectories,
    # so at +0d all later columns are NaN (which it rarely saw all-NaN simultaneously during training!).
    # Augmenting training with causal partial prefixes (k=0..4) teaches the model optimal early-epoch inference!
    print('Building causal prefix-augmented training set (teaches early +0d/+3d/+14d inference)...')
    aug_X_enr_list, aug_y_list, aug_w_list = [X_tr_enr], [y_train], [w_train * 2.0]  # weight full 2x, partials 0.5x each
    for k in range(len(TRAJECTORY_GRID) - 1):
        # Sample 40% of training events for each partial prefix to keep training fast and memory clean
        sub_idx = rng.choice(len(train_events), size=int(0.4 * len(train_events)), replace=False)
        partial_tr = [reveal_up_to(train_events[i], k, TRAJECTORY_GRID) for i in sub_idx]
        Xk, _ = extract_enriched_features(partial_tr)
        aug_X_enr_list.append(Xk)
        aug_y_list.append(y_train[sub_idx])
        aug_w_list.append(w_train[sub_idx] * 0.5)
    X_tr_enr_aug = np.vstack(aug_X_enr_list)
    y_train_aug = np.concatenate(aug_y_list)
    w_train_aug = np.concatenate(aug_w_list)

    models_info = []

    # 1. Baseline GBDT + Isotonic (Current Architecture)
    print('\n[1/6] Fitting Baseline GBDT (30 features, Isotonic calibration)...')
    t0 = time.time()
    m1_base = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1, max_iter=300, l2_regularization=1.0, random_state=0)
    m1 = CalibratedClassifierCV(m1_base, method='isotonic', cv=3).fit(X_tr_base, y_train, sample_weight=w_train)
    t1 = time.time() - t0
    s1_full = m1.predict_proba(X_te_base)[:, 1]
    s1_steps = {d: m1.predict_proba(X_te_base_steps[d])[:, 1] for d in TRAJECTORY_GRID}
    models_info.append(('1. Baseline GBDT (30 feat, Isotonic)', s1_full, s1_steps, t1, '#7f7f7f', '--'))

    # 2. Smooth-Calibrated Regularised GBDT (Platt Sigmoid + Larger Leaves)
    print('[2/6] Fitting Smooth-Calibrated GBDT (30 features, Sigmoid calibration + regularised leaves)...')
    t0 = time.time()
    m2_base = HistGradientBoostingClassifier(max_depth=5, min_samples_leaf=80, learning_rate=0.06, max_iter=350, l2_regularization=5.0, random_state=0)
    m2 = CalibratedClassifierCV(m2_base, method='sigmoid', cv=3).fit(X_tr_base, y_train, sample_weight=w_train)
    t2 = time.time() - t0
    s2_full = m2.predict_proba(X_te_base)[:, 1]
    s2_steps = {d: m2.predict_proba(X_te_base_steps[d])[:, 1] for d in TRAJECTORY_GRID}
    models_info.append(('2. Smooth-Calibrated GBDT (30 feat, Sigmoid)', s2_full, s2_steps, t2, '#1f77b4', '-.'))

    # 3. Physics-Enriched + Causal-Prefix GBDT (132 features: Snapshot + Fade + Broad Colours, Sigmoid)
    print('[3/6] Fitting Physics-Enriched + Prefix-Trained GBDT (132 features, Sigmoid)...')
    t0 = time.time()
    m3_base = HistGradientBoostingClassifier(max_depth=6, min_samples_leaf=80, learning_rate=0.06, max_iter=400, l2_regularization=5.0, random_state=0)
    m3 = CalibratedClassifierCV(m3_base, method='sigmoid', cv=3).fit(X_tr_enr_aug, y_train_aug, sample_weight=w_train_aug)
    t3 = time.time() - t0
    s3_full = m3.predict_proba(X_te_enr)[:, 1]
    s3_steps = {d: m3.predict_proba(X_te_enr_steps[d])[:, 1] for d in TRAJECTORY_GRID}
    models_info.append(('3. Physics-Enriched GBDT (Snap+Fade+Prefix)', s3_full, s3_steps, t3, '#ff7f0e', '-'))

    # 4. Smooth Spline GAM (B-Splines + Missingness Masks + Ridge Logistic)
    print('[4/6] Fitting Smooth Spline Logistic GAM (132 features, B-Splines + Masks)...')
    t0 = time.time()
    m4 = SplineGAMClassifier(n_knots=5, degree=3, C=0.5).fit(X_tr_enr_aug, y_train_aug, sample_weight=w_train_aug)
    t4 = time.time() - t0
    s4_full = m4.predict_proba(X_te_enr)
    s4_steps = {d: m4.predict_proba(X_te_enr_steps[d]) for d in TRAJECTORY_GRID}
    models_info.append(('4. Smooth Spline GAM (B-Splines + Masks)', s4_full, s4_steps, t4, '#2ca02c', ':'))

    # 5. Mask-Aware Smooth Neural Network Ensemble (MLP + Observation Masks)
    print('[5/6] Fitting Mask-Aware Smooth Neural MLP Ensemble (132 features + Masks)...')
    t0 = time.time()
    m5 = WeightedMaskMLPClassifier(n_estimators=4, n_samples=80000, hidden_layer_sizes=(128, 64, 32), alpha=2e-2).fit(
        X_tr_enr_aug, y_train_aug, sample_weight=w_train_aug
    )
    t5 = time.time() - t0
    s5_full = m5.predict_proba(X_te_enr)
    s5_steps = {d: m5.predict_proba(X_te_enr_steps[d]) for d in TRAJECTORY_GRID}
    models_info.append(('5. Mask-Aware Neural MLP Ensemble', s5_full, s5_steps, t5, '#9467bd', '--'))

    # 6. Hybrid Physics-Enriched GBDT + Smooth Neural MLP Ensemble (Log-Odds Blend)
    print('[6/6] Computing Hybrid Smooth Ensemble (Enriched GBDT + Neural MLP in Log-Odds)...')
    def blend_probs(p_a, p_b, w_a=0.65):
        pa = np.clip(p_a, 1e-7, 1 - 1e-7)
        pb = np.clip(p_b, 1e-7, 1 - 1e-7)
        logit = w_a * np.log(pa / (1 - pa)) + (1 - w_a) * np.log(pb / (1 - pb))
        return 1.0 / (1.0 + np.exp(-logit))

    s6_full = blend_probs(s3_full, s5_full, w_a=0.65)
    s6_steps = {d: blend_probs(s3_steps[d], s5_steps[d], w_a=0.65) for d in TRAJECTORY_GRID}
    models_info.append(('6. Hybrid Ensemble (Enriched GBDT + Smooth MLP)', s6_full, s6_steps, t3 + t5, '#d62728', '-'))

    # Evaluate and print comprehensive summary table
    summary = {'target': args.target, 'signal_label': signal_label, 'models': {}}
    print('\n' + '=' * 120)
    print(f'FULL-TRAJECTORY ARCHITECTURE BENCHMARK ({signal_label.upper()} vs UNLENSED BACKGROUND)')
    print('=' * 120)
    header = f'{"Architecture":<47} | {"AUC":<6} | {"TPR@10%":<7} | {"TPR@1%":<7} | {"TPR@0.1%":<8} | {"TPR@0.01%":<9} | {"Zero-FP TPR":<11} | {"Pure/yr":<7}'
    print(header)
    print('-' * 120)

    for name, s_full, s_steps, fit_time, color, ls in models_info:
        res_full = evaluate_scores(y_test, w_test, s_full, N_sig_ann, N_bg_ann)
        step_aucs = {}
        step_tpr1 = {}
        for d in TRAJECTORY_GRID:
            r_d = evaluate_scores(y_test, w_test, s_steps[d], N_sig_ann, N_bg_ann)
            step_aucs[str(int(d))] = r_d['auc']
            step_tpr1[str(int(d))] = r_d['tpr_1pct']

        print(
            f'{name:<47} | {res_full["auc"]:6.4f} | {res_full["tpr_10pct"]*100:6.2f}% | '
            f'{res_full["tpr_1pct"]*100:6.2f}% | {res_full["tpr_01pct"]*100:7.2f}% | '
            f'{res_full["tpr_001pct"]*100:8.2f}% | {res_full["zero_fp_tpr"]*100:10.2f}% | '
            f'{res_full["zero_fp_tp_per_yr"]:6.1f}'
        )

        summary['models'][name] = {
            'fit_time_sec': round(fit_time, 2),
            'auc_full': res_full['auc'],
            'tpr_at_10pct_fpr': res_full['tpr_10pct'],
            'tpr_at_1pct_fpr': res_full['tpr_1pct'],
            'tpr_at_01pct_fpr': res_full['tpr_01pct'],
            'tpr_at_001pct_fpr': res_full['tpr_001pct'],
            'zero_fp_tpr': res_full['zero_fp_tpr'],
            'zero_fp_tp_per_yr': res_full['zero_fp_tp_per_yr'],
            'zero_fp_raw_count': res_full['zero_fp_raw_count'],
            'auc_by_day': step_aucs,
            'tpr_1pct_by_day': step_tpr1,
            '_fpr': res_full['fpr_curve'],
            '_tpr': res_full['tpr_curve'],
        }
    print('=' * 120)

    print('\nEARLY-EPOCH PERFORMANCE (ROC AUC & TPR @ 1% FPR ACROSS +0d -> +60d):')
    print(f'{"Architecture":<47} | {"+0d AUC":<8} | {"+3d AUC":<8} | {"+14d AUC":<8} | {"+30d AUC":<8} | {"+60d AUC":<8}')
    print('-' * 98)
    for name, _, _, _, _, _ in models_info:
        sa = summary['models'][name]['auc_by_day']
        print(f'{name:<47} | {sa["0"]:8.4f} | {sa["3"]:8.4f} | {sa["14"]:8.4f} | {sa["30"]:8.4f} | {sa["60"]:8.4f}')
    print('-' * 98)

    # -------------------------------------------------------------------------
    # Plot 3-Panel Architecture Comparison Figure
    # -------------------------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5.8))

    # Panel 1: Log-FPR ROC Curve (Focusing on the Low-FPR LSST Alert Regime: 10^-4 to 1)
    for name, _, _, _, color, ls in models_info:
        mdata = summary['models'][name]
        fpr_c = mdata['_fpr']
        tpr_c = mdata['_tpr']
        lw = 2.6 if ('3.' in name or '6.' in name) else 1.8
        # Plot down to FPR = 5e-5; mark Zero-FP sweet spot on left axis
        pos = fpr_c > 0
        ax1.semilogx(fpr_c[pos] * 100, tpr_c[pos] * 100, color=color, ls=ls, lw=lw,
                     label=f'{name.split("(")[0].strip()} (AUC={mdata["auc_full"]:.3f})')
        ax1.scatter([0.004], [mdata['zero_fp_tpr'] * 100], color=color, marker='*', s=130, edgecolor='black', lw=0.6, zorder=5)

    ax1.set_xlim(0.003, 100)
    ax1.set_ylim(0, 102)
    ax1.set_xlabel('False Positive Rate (FPR, % — Log Scale; ★ = 0% FPR Sweet Spot)', fontsize=10.5)
    ax1.set_ylabel('Lensed Recovery (TPR, %)', fontsize=10.5)
    ax1.set_title(f'A. Low-FPR ROC Comparison ({signal_label})\nUltra-Low Contamination Regime', fontsize=11.5, fontweight='bold')
    ax1.grid(True, which='both', linestyle='--', alpha=0.45)
    ax1.legend(loc='lower right', fontsize=8.2)

    # Panel 2: Temporal Evolution of ROC AUC (+0d to +60d)
    days = [int(d) for d in TRAJECTORY_GRID]
    for name, _, _, _, color, ls in models_info:
        mdata = summary['models'][name]
        ys = [mdata['auc_by_day'][str(d)] for d in days]
        lw = 2.6 if ('3.' in name or '6.' in name) else 1.8
        ax2.plot(days, ys, color=color, ls=ls, lw=lw, marker='o', ms=6, label=name.split('(')[0].strip())

    ax2.set_xlabel('Days Since First LSST Detection (Δt)', fontsize=10.5)
    ax2.set_ylabel('Held-Out Rate-Weighted ROC AUC', fontsize=10.5)
    ax2.set_title('B. Early-to-Late Trajectory Discrimination\nImpact of Causal-Prefix & Fade Features', fontsize=11.5, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.45)
    ax2.legend(loc='lower right', fontsize=8.2)

    # Panel 3: LSST Purity vs Annual True Lensed Yield
    for name, _, _, _, color, ls in models_info:
        mdata = summary['models'][name]
        fpr_c = mdata['_fpr']
        tpr_c = mdata['_tpr']
        tp_yr = tpr_c * N_sig_ann
        fp_yr = fpr_c * N_bg_ann
        with np.errstate(divide='ignore', invalid='ignore'):
            purity = np.where((tp_yr + fp_yr) > 0, tp_yr / (tp_yr + fp_yr) * 100.0, 100.0)
        valid = tpr_c > 0
        lw = 2.6 if ('3.' in name or '6.' in name) else 1.8
        ax3.plot(tp_yr[valid], purity[valid], color=color, ls=ls, lw=lw, label=name.split('(')[0].strip())
        ax3.scatter([mdata['zero_fp_tp_per_yr']], [100.0], color=color, marker='*', s=140, edgecolor='black', lw=0.6, zorder=5)

    ax3.set_xlabel(f'Recovered Lensed SNe per Year (out of {N_sig_ann:.0f}/yr)', fontsize=10.5)
    ax3.set_ylabel('Sample Purity: TP / (TP + FP) (%)', fontsize=10.5)
    ax3.set_title(f'C. Real LSST Survey Purity vs Annual Yield\n(1 : {int(N_bg_ann/N_sig_ann):,} Base-Rate Prior Applied)', fontsize=11.5, fontweight='bold')
    ax3.set_xlim(0, N_sig_ann * 0.55)
    ax3.set_ylim(-2, 105)
    ax3.grid(True, linestyle='--', alpha=0.45)
    ax3.legend(loc='upper right', fontsize=8.2)

    fig.tight_layout()
    plot_path = os.path.join(args.out_dir, f'trajectory_model_comparison{suffix}.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {plot_path}')

    # Strip large numpy arrays before saving JSON
    for mdata in summary['models'].values():
        del mdata['_fpr']
        del mdata['_tpr']
    json_path = os.path.join(args.out_dir, f'trajectory_model_comparison{suffix}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote {json_path}')


if __name__ == '__main__':
    main()
