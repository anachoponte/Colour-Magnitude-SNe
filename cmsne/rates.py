"""Absolute rate calculation framework adapted from Lensed Transient Toolkit (ltt).

Provides physical volumetric rate models, differential comoving volume integration,
and differential lensing optical depth for calculating absolute detection rates
(events/year) in survey footprints like LSST WFD.
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import quad
from astropy.cosmology import Planck18 as cosmo

from .lsst import CC_FRACTIONS

# LSST Wide-Fast-Deep survey footprint
LSST_SURVEY_AREA_DEG2 = 18000.0
LSST_SURVEY_SOLID_ANGLE = LSST_SURVEY_AREA_DEG2 * (np.pi / 180.0) ** 2  # ~5.483 sr

# Canonical local volumetric rates (events / Mpc^3 / yr)
# SN Ia: Frohmaier et al. (2019), Dilday et al. (2010)
SN_IA_LOCAL_RATE = 2.4e-5
# Core-Collapse: Li et al. (2011), Taylor et al. (2014)
CC_LOCAL_RATE = 7.0e-5


class SourceRateDistribution(ABC):
    """Abstract base class for transient source rate distributions."""

    @abstractmethod
    def rate_density(self, redshift):
        """Volumetric rate density at given redshift(s) [events/Mpc^3/yr]."""
        pass

    @abstractmethod
    def get_local_rate(self):
        """Local (z=0) volumetric rate density [events/Mpc^3/yr]."""
        pass


class MadauDickinsonSFRD(SourceRateDistribution):
    """Source rate distribution following Madau-Dickinson SFRD evolution.

    The volumetric rate follows:
        R(z) = R0 * g(z) / g(0)
    where
        g(z) = (1+z)^alpha * [1 + ((1+z)/(1+z_pivot))^(alpha+beta)]^(-1)
    """

    def __init__(self, local_rate, alpha=2.7, beta=2.9, z_pivot=1.9):
        self.local_rate = float(local_rate)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.z_pivot = float(z_pivot)
        self._norm = self._g(0.0)

    def _g(self, z):
        z_ratio = (1.0 + z) / (1.0 + self.z_pivot)
        return (1.0 + z) ** self.alpha * (1.0 + z_ratio ** (self.alpha + self.beta)) ** (-1)

    def rate_density(self, redshift):
        return self.local_rate * self._g(redshift) / self._norm

    def get_local_rate(self):
        return self.local_rate


class CustomRateDistribution(SourceRateDistribution):
    """Custom source rate distribution from a user-provided callable f(z)."""

    def __init__(self, rate_function, local_rate):
        self.rate_function = rate_function
        self.local_rate = float(local_rate)

    def rate_density(self, redshift):
        return self.rate_function(redshift)

    def get_local_rate(self):
        return self.local_rate


class LensingOpticalDepth:
    """Calculate optical depth to strong gravitational lensing.

    Differential optical depth:
        dtau/dmu dz = (D_com(z) / 31 Gpc)^3 * mu^(-3)
    """

    @staticmethod
    def dtau_dmu_dz(redshift, magnification):
        """Calculate differential optical depth dtau/dmu dz."""
        d_com = cosmo.comoving_distance(redshift).value  # Mpc
        d_gpc = d_com / 1e3
        return (d_gpc / 31.0) ** 3 * (magnification ** -3)

    @staticmethod
    def tau(redshift, mu_min=2.0, mu_max=100.0):
        """Total optical depth integrated over magnification range."""
        if np.isscalar(redshift):
            res, _ = quad(lambda mu: LensingOpticalDepth.dtau_dmu_dz(redshift, mu),
                          mu_min, mu_max)
            return res
        return np.array([LensingOpticalDepth.tau(z, mu_min, mu_max) for z in redshift])


def check_images_resolved(image_separation, seeing):
    """Check if lensed images are spatially resolved."""
    return image_separation > seeing


class DetectionMatrix:
    """Multi-dimensional matrix tracking detections as a function of parameters."""

    def __init__(self, parameter_names, parameter_bins):
        self.parameter_names = list(parameter_names)
        self.parameter_bins = [np.asarray(b) for b in parameter_bins]
        self.n_params = len(self.parameter_names)
        shape = tuple(len(b) - 1 for b in self.parameter_bins)
        self.detection_matrix = np.zeros(shape, dtype=float)
        self.n_trials_matrix = np.zeros(shape, dtype=float)
        self.bin_centers = [0.5 * (b[:-1] + b[1:]) for b in self.parameter_bins]

    def add_event(self, parameter_values, detected=True):
        if isinstance(parameter_values, dict):
            values = [parameter_values[name] for name in self.parameter_names]
        else:
            values = list(parameter_values)
        indices = []
        for val, bins in zip(values, self.parameter_bins):
            idx = np.digitize(val, bins) - 1
            if idx < 0 or idx >= len(bins) - 1:
                return
            indices.append(idx)
        indices = tuple(indices)
        self.n_trials_matrix[indices] += 1
        if detected:
            self.detection_matrix[indices] += 1

    def get_detection_efficiency(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            eff = self.detection_matrix / self.n_trials_matrix
            eff[~np.isfinite(eff)] = 0.0
        return eff


class RateCalculator:
    """Calculate detection rates for transient populations in a survey footprint."""

    def __init__(self, rate_distribution, survey_area=LSST_SURVEY_AREA_DEG2,
                 survey_duration=1.0):
        self.rate_distribution = rate_distribution
        self.survey_area = float(survey_area)
        self.survey_duration = float(survey_duration)
        self.survey_solid_angle = self.survey_area * (np.pi / 180.0) ** 2

    def source_rate_density(self, z):
        """Source rate per unit redshift per year in survey footprint [yr^-1].

        f(z) = R(z) * (dV/dOmega dz) * Omega * (1+z)^(-1)
        """
        r_z = self.rate_distribution.rate_density(z)
        dv_dz = cosmo.differential_comoving_volume(z).value  # Mpc^3/sr
        return r_z * dv_dz * self.survey_solid_angle * ((1.0 + z) ** -1)

    def sources_in_redshift_bin(self, z_min, z_max, use_integration=True):
        """Expected unlensed sources per year in redshift bin [z_min, z_max]."""
        if use_integration:
            res, _ = quad(self.source_rate_density, z_min, z_max, limit=200, epsabs=1e-5)
            return res
        z_mean = 0.5 * (z_min + z_max)
        r_mean = self.rate_distribution.rate_density(z_mean)
        v_min = cosmo.comoving_volume(z_min).value
        v_max = cosmo.comoving_volume(z_max).value
        delta_v = v_max - v_min
        return (r_mean * delta_v * (self.survey_solid_angle / (4 * np.pi)) *
                ((1.0 + z_mean) ** -1))

    def calculate_from_simulations(self, transients, parameter_bins,
                                   detection_threshold=1):
        """Calculate detection rates from simulated transients using Monte Carlo."""
        from scipy.stats import binned_statistic_dd

        param_names = list(parameter_bins.keys())
        edges = [parameter_bins[name] for name in param_names]
        if 'redshift' not in param_names:
            raise ValueError("Must include 'redshift' in parameter_bins")

        data, weights, is_detected = [], [], []
        for t in transients:
            pv = [getattr(t, name) if hasattr(t, name) else t[name] for name in param_names]
            data.append(pv)
            w = getattr(t, 'weight', 1.0) if hasattr(t, 'weight') else t.get('weight', 1.0)
            weights.append(w)
            det = False
            if hasattr(t, 'detections'):
                d = t.detections
                det = sum(d.values()) >= detection_threshold if isinstance(d, dict) else d >= detection_threshold
            elif isinstance(t, dict):
                det = t.get('detected', True)
            is_detected.append(det)

        data = np.asarray(data)
        weights = np.asarray(weights, dtype=float)
        is_detected = np.asarray(is_detected, dtype=bool)

        z_idx = param_names.index('redshift')
        z_vals = data[:, z_idx]
        f_z = np.array([self.source_rate_density(z) for z in z_vals])
        full_weights = f_z * weights
        det_full_weights = full_weights * is_detected

        volume_factor = np.prod([edge[-1] - edge[0] for edge in edges])
        n_total = len(transients)

        stat_sum, bin_edges, _ = binned_statistic_dd(
            data, det_full_weights, bins=edges, statistic='sum'
        )
        counts_unweighted, _, _ = binned_statistic_dd(
            data, weights, bins=edges, statistic='count'
        )
        stat_scaled = (volume_factor / n_total) * stat_sum
        total_rate = (volume_factor / n_total) * float(np.sum(det_full_weights))

        det_counts, _, _ = binned_statistic_dd(
            data, is_detected.astype(float), bins=edges, statistic='sum'
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            eff = det_counts / counts_unweighted
            eff = np.nan_to_num(eff, nan=0.0)

        bin_centers = [0.5 * (be[:-1] + be[1:]) for be in bin_edges]
        return {
            'statistic': stat_scaled,
            'bin_edges': bin_edges,
            'bin_centers': bin_centers,
            'bin_names': param_names,
            'counts': counts_unweighted,
            'efficiency': eff,
            'total_rate': total_rate,
        }

    def marginalize(self, results, keep_parameters, normalize=False):
        """Marginalize detection rates over unwanted parameters."""
        if isinstance(keep_parameters, str):
            keep_parameters = [keep_parameters]
        param_names = results['bin_names']
        stat = results['statistic']
        bin_centers = results['bin_centers']
        keep_axes = [param_names.index(name) for name in keep_parameters]
        sum_axes = tuple(i for i in range(len(param_names)) if i not in keep_axes)
        marg = np.sum(stat, axis=sum_axes)
        if normalize and np.sum(marg) > 0:
            marg = marg / np.sum(marg)
        if len(keep_parameters) == 1:
            centers = bin_centers[keep_axes[0]]
        else:
            centers = [bin_centers[i] for i in keep_axes]
        return centers, marg


# Default rate models and calculators for LSST
_sn_ia_rate_dist = None
_sn_cc_rate_dist = None
_sn_ia_calc = None
_sn_cc_calc = None


def _compute_dtd_sn_ia_interpolator():
    """Calibrated DTD-convolved SN Ia volumetric rate density [events/Mpc^3/yr]."""
    from scipy.interpolate import interp1d
    from .sn_rates import rate_grid
    z_grid, raw_grid = rate_grid()
    # Normalize to canonical local rate SN_IA_LOCAL_RATE at z=0.05
    i0 = int(np.argmin(np.abs(z_grid - 0.05)))
    scale = SN_IA_LOCAL_RATE / raw_grid[i0]
    return interp1d(z_grid, raw_grid * scale, kind='cubic', fill_value='extrapolate')


def sn_ia_rate_distribution(model='dtd'):
    """Canonical SN Ia rate distribution.

    :param model: ``'dtd'`` (default) for delay-time-distribution convolved rate
                  normalized to local rate 2.4e-5 Mpc^-3 yr^-1; ``'sfrd'`` for
                  MadauDickinsonSFRD.
    """
    global _sn_ia_rate_dist
    if _sn_ia_rate_dist is None:
        if model == 'dtd':
            f_dtd = _compute_dtd_sn_ia_interpolator()
            _sn_ia_rate_dist = CustomRateDistribution(
                rate_function=lambda z: np.maximum(f_dtd(z), 0.0),
                local_rate=SN_IA_LOCAL_RATE
            )
        else:
            _sn_ia_rate_dist = MadauDickinsonSFRD(local_rate=SN_IA_LOCAL_RATE,
                                                  alpha=2.7, beta=2.9, z_pivot=1.9)
    return _sn_ia_rate_dist


def sn_cc_rate_distribution():
    """Canonical core-collapse rate distribution (MadauDickinsonSFRD, R0=7.0e-5)."""
    global _sn_cc_rate_dist
    if _sn_cc_rate_dist is None:
        _sn_cc_rate_dist = MadauDickinsonSFRD(local_rate=CC_LOCAL_RATE,
                                              alpha=2.7, beta=2.9, z_pivot=1.9)
    return _sn_cc_rate_dist


def sn_ia_rate_calculator():
    """Canonical SN Ia RateCalculator for LSST WFD."""
    global _sn_ia_calc
    if _sn_ia_calc is None:
        _sn_ia_calc = RateCalculator(sn_ia_rate_distribution(),
                                     survey_area=LSST_SURVEY_AREA_DEG2)
    return _sn_ia_calc


def sn_cc_rate_calculator():
    """Canonical core-collapse RateCalculator for LSST WFD."""
    global _sn_cc_calc
    if _sn_cc_calc is None:
        _sn_cc_calc = RateCalculator(sn_cc_rate_distribution(),
                                     survey_area=LSST_SURVEY_AREA_DEG2)
    return _sn_cc_calc


def absolute_rate_weight(rate_kind, z, magnification=1.0, source=None,
                         z_range=None, magnification_range=None, n_attempted=None):
    """Calculate absolute rate weight for a simulated transient.

    If ``z_range`` and ``n_attempted`` are provided, returns the exact Monte Carlo
    weight contributing directly to the annual detection rate [events/year] in LSST:
        w_i = (V_sample / N_attempted) * differential_rate_density

    If ``n_attempted`` is None, returns the bare differential rate density:
        f(z) for unlensed, or f(z) * dtau/dmu_dz for lensed.

    :param rate_kind: ``'uIa'`` (unlensed Ia), ``'sig'`` (lensed Ia),
                      ``'uCC'`` (unlensed CC), ``'lCC'`` (lensed CC)
    :param z: redshift
    :param magnification: magnification (default 1.0)
    :param source: sncosmo source name (used for core-collapse sub-type fractions)
    :param z_range: tuple (z_min, z_max) sampling range
    :param magnification_range: tuple (mu_min, mu_max) sampling range
    :param n_attempted: total simulated attempts for this class
    :return: float rate weight
    """
    calc_ia = sn_ia_rate_calculator()
    calc_cc = sn_cc_rate_calculator()

    is_lensed = (rate_kind in ('sig', 'lCC'))
    is_cc = (rate_kind in ('uCC', 'lCC'))

    calc = calc_cc if is_cc else calc_ia
    f_z = calc.source_rate_density(z)

    # Sub-type volumetric fraction for core-collapse templates
    sub_frac = CC_FRACTIONS.get(source, 1.0) if is_cc else 1.0

    if is_lensed:
        dtau = LensingOpticalDepth.dtau_dmu_dz(z, magnification)
        density = f_z * sub_frac * dtau
    else:
        density = f_z * sub_frac

    if n_attempted is not None and z_range is not None:
        delta_z = float(z_range[1] - z_range[0])
        if is_lensed:
            mu_r = magnification_range if magnification_range is not None else (2.0, 50.0)
            delta_mu = float(mu_r[1] - mu_r[0])
            v_sample = delta_z * delta_mu
        else:
            v_sample = delta_z
        return float((v_sample / n_attempted) * density)

    return float(density)
