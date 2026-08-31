"""LSST band properties and plotting styles.

Holds the per-band zeropoints / seeing / sky-brightness / limiting magnitudes
(:class:`LSSTproperties`), the plotting colour/marker maps, and the single-visit
detection limits used by the colour-magnitude analysis.
"""

# Plotting styles keyed by sncosmo LSST band name.
colours = {'lsstu': 'b', 'lsstg': 'c', 'lsstr': 'g',
           'lssti': 'orange', 'lsstz': 'r', 'lssty': 'm'}

markers = {'lsstu': 'x', 'lsstg': '>', 'lsstr': '<',
           'lssti': '^', 'lsstz': 'o', 'lssty': 's'}

bands = ['lsstu', 'lsstg', 'lsstr', 'lssti', 'lsstz', 'lssty']

# Band pairs used for the colour-magnitude diagram grid, bluer band first.
band_pairs = [('lsstg', 'lsstr'), ('lsstg', 'lssti'), ('lsstg', 'lsstz'),
              ('lsstg', 'lssty'), ('lsstr', 'lssti'), ('lsstr', 'lsstz'),
              ('lsstr', 'lssty'), ('lssti', 'lsstz'), ('lssti', 'lssty'),
              ('lsstz', 'lssty')]

# Core-collapse contaminants: (mean peak absolute magnitude, scatter, sncosmo source).
contaminant_info = [
    (-17.51, 0.74, 'nugent-sn1bc'),
    (-17.46, 0.38, 'nugent-sn2l'),
    (-19.05, 0.50, 'nugent-sn2n'),
    (-16.90, 1.12, 'nugent-sn2p'),
]

# Relative volumetric fractions of the core-collapse sub-types (of all CC), from
# volume-limited samples (Li et al. 2011; Shivvers et al. 2017). Used to weight the
# contaminant background by a realistic sub-type mix rather than equal-per-template.
# Sum to 1; feed them to the rate weight so, e.g., the rare SN IIn/IIL do not carry
# the same weight as the common SN IIP.
CC_FRACTIONS = {
    'nugent-sn1bc': 0.36,   # Ib / Ic / IIb
    'nugent-sn2p':  0.55,   # II-P (most common)
    'nugent-sn2n':  0.06,   # II-n (rare)
    'nugent-sn2l':  0.03,   # II-L (rare)
}

# Volumetric core-collapse : SN Ia rate ratio. The package's Ia and CC rate grids
# carry independent normalisations, so this constant sets the cross-class balance.
# Anchored to the local (z~0) literature rates -- SN Ia ~2.4e-5/yr/Mpc^3
# (Frohmaier et al. 2019) and CC ~7e-5/yr/Mpc^3 (Li et al. 2011; Taylor et al.
# 2014), i.e. CC:Ia ~= 3:1 -- which the grids' own z~0.05 ratio (~0.86) is scaled to
# reach (3 / 0.86 ~= 3.5). The redshift evolution of the ratio then follows the
# SFR (CC) and delay-time-distribution (Ia) shapes in cmsne.sn_rates.
CC_TO_IA_RATE = 3.5


class LSSTproperties:
    LSST_u = {'magnitude_zero_point': 26.52,  # from https://smtn-002.lsst.io/ check if up to date!!
              'average_seeing': 0.92,
              'sky_brightness': 23.05,
              'limiting_magnitude': 23.9}

    LSST_g = {'magnitude_zero_point': 28.51,
              'average_seeing': 0.87,
              'sky_brightness': 22.25,
              'limiting_magnitude': 25.0}

    LSST_r = {'magnitude_zero_point': 28.36,
              'average_seeing': 0.83,
              'sky_brightness': 21.2,
              'limiting_magnitude': 24.7}

    LSST_i = {'magnitude_zero_point': 28.17,
              'average_seeing': 0.80,
              'sky_brightness': 20.46,
              'limiting_magnitude': 24.0}

    LSST_z = {'magnitude_zero_point': 27.78,
              'average_seeing': 0.78,
              'sky_brightness': 19.61,
              'limiting_magnitude': 23.3}

    LSST_y = {'magnitude_zero_point': 26.82,
              'average_seeing': 0.76,
              'sky_brightness': 18.6,
              'limiting_magnitude': 22.1}

    @classmethod
    def get_properties(cls, band):
        if band == 'lsstu':
            return cls.LSST_u
        elif band == 'lsstg':
            return cls.LSST_g
        elif band == 'lsstr':
            return cls.LSST_r
        elif band == 'lssti':
            return cls.LSST_i
        elif band == 'lsstz':
            return cls.LSST_z
        elif band == 'lssty':
            return cls.LSST_y
        else:
            raise ValueError("band %s not supported! Choose 'lsstu', 'lsstg', 'lsstr', 'lssti', 'lsstz' or 'lssty' for LSST." % band)


def mask(band):
    """Single-visit limiting magnitude used as a detection threshold per band."""
    if band == 'lsstu':
        return 23.9
    elif band == 'lsstg':
        return 25.0
    elif band == 'lsstr':
        return 24.7
    elif band == 'lssti':
        return 24.0
    elif band == 'lsstz':
        return 23.3
    elif band == 'lssty':
        return 22.1
