"""Detection-aware importance-sampling proposals.

Every simulated population in this project is drawn from some proposal over the
latent variables ``(z, mu)`` and reweighted by the physical rate.  Historically
that proposal was *uniform*, which is a poor match to a target whose lensed
component falls as ``mu^-3``: effective sample size runs at 0.6-2.9% of the raw
event count for signal pools, with the top 1% of events carrying 55-62% of all
weight.

The variance-optimal proposal here is **not** the one matched to the rate.
Detection is a random 0/1 outcome of simulating a light curve rather than a
function we can evaluate, which puts this in the *noisy* importance-sampling
regime.  For a noisy integrand evaluation with mean ``m`` and standard
deviation ``s``, the optimal proposal is ``q ∝ ||f|| * sqrt(m^2 + s^2)``
(Llorente & Martino 2025, arXiv:2502.07396, section 6.2).  A Bernoulli
detection indicator has ``m = eps`` and ``s^2 = eps(1 - eps)``, so
``m^2 + s^2 = eps`` and

    q_opt(z, mu)  ∝  rate(z, mu) * sqrt(eps(z, mu))

Equivalently this is Neyman (1934) allocation: sample in proportion to standard
deviation, not to size.  Note that ``rate * eps`` -- the answer the standard
noise-free result ``q ∝ |f| p`` would give -- over-corrects, and plain ``rate``
*under*-corrects so badly that it loses to uniform for most classes here.

``eps`` is unknown a priori and must be estimated from a pilot stage, which
makes it noisy.  A badly estimated efficiency can make importance sampling
*worse* than uniform (the known failure mode of Neyman allocation).  Three
layers guard against that, the last of which is not optional:

1. Beta shrinkage of the per-cell ``detected / attempted`` ratio.
2. Optional neighbour smoothing across the grid.
3. A **defensive mixture** with the uniform proposal
   (Hesterberg 1995; Owen & Zhou 2000), which bounds every weight at
   ``1 / alpha`` times its uniform-proposal value.

The proposal is represented as a piecewise-constant density on a 2-D grid.
``eps`` is *not* separable in ``(z, mu)`` -- detection depends on apparent
brightness, which couples them -- so a product of two 1-D proposals would be
wrong.  A piecewise-constant grid also makes ``density()`` exact, so no
interpolation error leaks into the importance weights.
"""

import numpy as np

__all__ = ['DetectionAwareProposal']

#: Default defensive-mixture weight on the uniform component.
DEFAULT_ALPHA = 0.1
#: Default floor applied to the estimated efficiency after smoothing.
DEFAULT_EPS_MIN = 1e-3
#: Default Beta(a, b) shrinkage prior for per-cell detection fractions.
DEFAULT_EPS_PRIOR = (0.5, 0.5)


class DetectionAwareProposal:
    """Piecewise-constant proposal over ``(z, mu)`` matched to ``rate * sqrt(eps)``.

    :param rate_fn: callable ``rate_fn(z, mu) -> array``, the *same* function
        whose value becomes the importance weight numerator.  Passing the
        project's own rate function rather than restating ``mu^-3`` here is
        deliberate: it keeps the physics in one place.
    :param z_range: ``(z_min, z_max)`` sampling range for this class.
    :param magnification_range: ``(mu_min, mu_max)``; ignored when not lensed.
    :param lensed: when ``False``, ``mu`` is fixed at 1 and the density is with
        respect to ``dz`` alone, mirroring ``absolute_rate_weight``'s own
        ``v_sample = delta_z`` convention.
    :param n_z, n_mu: grid resolution.  ``mu`` cells are log-spaced, since the
        rate varies over orders of magnitude across the magnification range.
    :param alpha: defensive-mixture weight on the uniform component.  Must be
        in ``[0, 1)``.  Zero disables the safety net and is intended for tests
        only.
    :param eps_min: floor applied to the efficiency after smoothing.
    :param eps_prior: ``(a, b)`` Beta shrinkage prior for ``detected/attempted``.
    """

    def __init__(self, rate_fn, z_range, magnification_range=(2.0, 50.0),
                 lensed=True, n_z=48, n_mu=48, alpha=DEFAULT_ALPHA,
                 eps_min=DEFAULT_EPS_MIN, eps_prior=DEFAULT_EPS_PRIOR,
                 smooth_passes=1):
        if not (0.0 <= alpha < 1.0):
            raise ValueError('alpha must be in [0, 1), got %r' % (alpha,))
        if not (0.0 < eps_min <= 1.0):
            raise ValueError('eps_min must be in (0, 1], got %r' % (eps_min,))

        self.rate_fn = rate_fn
        self.lensed = bool(lensed)
        self.alpha = float(alpha)
        self.eps_min = float(eps_min)
        self.eps_prior = tuple(eps_prior)
        self.smooth_passes = int(smooth_passes)

        self.z_range = (float(z_range[0]), float(z_range[1]))
        self.z_edges = np.linspace(self.z_range[0], self.z_range[1], int(n_z) + 1)

        if self.lensed:
            mu_lo, mu_hi = float(magnification_range[0]), float(magnification_range[1])
            if mu_lo <= 0:
                raise ValueError('magnification range must be positive for log spacing')
            self.magnification_range = (mu_lo, mu_hi)
            # Log spacing: the rate falls as mu^-3, so linear cells would spend
            # almost all their resolution where the rate is negligible.
            self.mu_edges = np.geomspace(mu_lo, mu_hi, int(n_mu) + 1)
        else:
            self.magnification_range = None
            # One degenerate mu cell of unit width keeps the array algebra
            # uniform; the density is reported per dz only (see `density`).
            self.mu_edges = np.array([1.0, 2.0])

        self.z_centres = 0.5 * (self.z_edges[1:] + self.z_edges[:-1])
        # Geometric centres for log-spaced mu cells.
        self.mu_centres = np.sqrt(self.mu_edges[1:] * self.mu_edges[:-1])

        self.cell_area = np.outer(np.diff(self.z_edges), np.diff(self.mu_edges))
        self.total_area = float(self.cell_area.sum())

        ZC, MC = np.meshgrid(self.z_centres, self.mu_centres, indexing='ij')
        rate = np.asarray(rate_fn(ZC.ravel(), MC.ravel()), dtype=float).reshape(ZC.shape)
        if not np.all(np.isfinite(rate)):
            raise ValueError('rate_fn produced non-finite values on the grid')
        self.rate_grid = np.clip(rate, 0.0, None)

        # Until a pilot runs, assume uniform efficiency: the proposal then
        # reduces to the rate-matched one, which is still unbiased.
        self.eps_hat = np.ones_like(self.rate_grid)
        self._rebuild()

    # ------------------------------------------------------------------ build

    def _rebuild(self):
        """Recompute cell probabilities from `rate_grid` and `eps_hat`."""
        matched = self.rate_grid * np.sqrt(self.eps_hat) * self.cell_area
        total = float(matched.sum())
        if total > 0:
            matched = matched / total
        else:
            # Degenerate: no rate anywhere on the grid. Fall back to uniform so
            # the object stays a valid distribution instead of raising.
            matched = self.cell_area / self.total_area

        uniform = self.cell_area / self.total_area
        self.cell_prob = (1.0 - self.alpha) * matched + self.alpha * uniform
        self.cell_prob = self.cell_prob / self.cell_prob.sum()

        self._flat_cdf = np.cumsum(self.cell_prob.ravel())
        self._flat_cdf[-1] = 1.0
        # Piecewise-constant density: probability per unit area in each cell.
        self._density_grid = self.cell_prob / self.cell_area

    # ------------------------------------------------------------- efficiency

    def set_efficiency_grid(self, eps):
        """Set ``eps_hat`` directly from an array shaped like the grid.

        Values are floored at ``eps_min`` so that no cell with non-zero rate
        can end up with zero proposal mass.  Mainly for tests and for reusing a
        previously measured efficiency.
        """
        eps = np.asarray(eps, dtype=float)
        if eps.shape != self.rate_grid.shape:
            raise ValueError('efficiency grid has shape %r, expected %r'
                             % (eps.shape, self.rate_grid.shape))
        if not np.all(np.isfinite(eps)):
            raise ValueError('efficiency grid contains non-finite values')
        self.eps_hat = np.clip(eps, self.eps_min, 1.0)
        self._rebuild()
        return self

    def estimate_efficiency(self, z, mu, detected):
        """Estimate ``eps`` from pilot outcomes and rebuild the proposal.

        :param z, mu: latent values of every pilot **attempt** (not only the
            detected ones -- the denominator is what makes this an efficiency).
        :param detected: boolean array, whether each attempt passed the
            detection cut.

        Per cell this forms the Beta-shrunk fraction ``(d + a) / (n + a + b)``,
        optionally smooths across neighbouring cells, then applies the
        ``eps_min`` floor.  Shrinkage matters because thinly populated cells
        would otherwise report 0 or 1 from a handful of attempts and steer the
        proposal on noise.
        """
        z = np.asarray(z, dtype=float).ravel()
        mu = np.asarray(mu, dtype=float).ravel()
        detected = np.asarray(detected).astype(bool).ravel()
        if not (z.size == mu.size == detected.size):
            raise ValueError('z, mu and detected must have equal length')
        if z.size == 0:
            raise ValueError('cannot estimate efficiency from zero attempts')

        iz, imu = self._cell_index(z, mu)
        shape = self.rate_grid.shape
        flat = iz * shape[1] + imu
        n_att = np.bincount(flat, minlength=shape[0] * shape[1]).reshape(shape)
        n_det = np.bincount(flat, weights=detected.astype(float),
                            minlength=shape[0] * shape[1]).reshape(shape)

        a, b = self.eps_prior
        eps = (n_det + a) / (n_att + a + b)

        for _ in range(max(0, self.smooth_passes)):
            eps = _smooth3(eps, weights=n_att)

        self.eps_hat = np.clip(eps, self.eps_min, 1.0)
        self.n_attempted_grid = n_att
        self._rebuild()
        return self

    def efficiency_at(self, z, mu):
        """Estimated detection efficiency at the given points."""
        iz, imu = self._cell_index(np.asarray(z, float).ravel(),
                                   np.asarray(mu, float).ravel())
        return self.eps_hat[iz, imu]

    # ---------------------------------------------------------- sample/density

    def _cell_index(self, z, mu):
        iz = np.clip(np.searchsorted(self.z_edges, z, side='right') - 1,
                     0, len(self.z_centres) - 1)
        if self.lensed:
            imu = np.clip(np.searchsorted(self.mu_edges, mu, side='right') - 1,
                          0, len(self.mu_centres) - 1)
        else:
            imu = np.zeros_like(iz)
        return iz, imu

    def sample(self, n, rng=None):
        """Draw ``n`` latent pairs.  Returns ``(z, mu)`` arrays.

        A cell is chosen from the discrete CDF, then a point uniformly within
        it, so the realised density is exactly ``density()``.
        """
        n = int(n)
        if rng is None:
            rng = np.random.default_rng()
        elif isinstance(rng, (int, np.integer)):
            rng = np.random.default_rng(rng)

        flat = np.searchsorted(self._flat_cdf, rng.random(n), side='right')
        flat = np.clip(flat, 0, self.cell_prob.size - 1)
        iz, imu = np.unravel_index(flat, self.cell_prob.shape)

        z = self.z_edges[iz] + rng.random(n) * np.diff(self.z_edges)[iz]
        if self.lensed:
            mu = self.mu_edges[imu] + rng.random(n) * np.diff(self.mu_edges)[imu]
        else:
            mu = np.ones(n)
        return z, mu

    def density(self, z, mu):
        """Proposal density at the given points.

        With respect to ``dz dmu`` for lensed classes and ``dz`` for unlensed
        ones -- matching ``absolute_rate_weight``'s own ``v_sample``
        convention, so that the uniform case reproduces ``1/(delta_z*delta_mu)``
        and ``1/delta_z`` respectively.

        Strictly positive everywhere in the domain whenever ``alpha > 0``.
        """
        z = np.asarray(z, dtype=float)
        mu = np.asarray(mu, dtype=float)
        iz, imu = self._cell_index(z.ravel(), mu.ravel())
        return self._density_grid[iz, imu].reshape(z.shape)

    # --------------------------------------------------------------- helpers

    @classmethod
    def uniform(cls, z_range, magnification_range=(2.0, 50.0), lensed=True, **kw):
        """A proposal identical to the historical uniform sampling.

        Useful as a pilot, as a baseline in ESS comparisons, and to show that
        the general weight formula reduces to the old one.
        """
        kw.setdefault('n_z', 1)
        kw.setdefault('n_mu', 1)
        kw['alpha'] = 0.0
        return cls(lambda z, mu: np.ones_like(np.asarray(z, float)),
                   z_range, magnification_range, lensed=lensed, **kw)

    def __repr__(self):
        kind = 'lensed' if self.lensed else 'unlensed'
        return ('DetectionAwareProposal(%s, z=%s, mu=%s, grid=%dx%d, alpha=%.3g)'
                % (kind, self.z_range, self.magnification_range,
                   self.rate_grid.shape[0], self.rate_grid.shape[1], self.alpha))


def _smooth3(grid, weights=None):
    """One pass of a 3x3 neighbour average, optionally weighted by counts.

    Weighting by attempt counts stops a thinly sampled cell from dragging a
    well-measured neighbour around.  Edges use available neighbours only.
    """
    grid = np.asarray(grid, dtype=float)
    w = np.ones_like(grid) if weights is None else np.asarray(weights, float) + 1.0
    num = np.zeros_like(grid)
    den = np.zeros_like(grid)
    for dz in (-1, 0, 1):
        for dm in (-1, 0, 1):
            g = np.roll(np.roll(grid, dz, axis=0), dm, axis=1)
            ww = np.roll(np.roll(w, dz, axis=0), dm, axis=1)
            mask = np.ones_like(grid, dtype=bool)
            if dz == 1:
                mask[0, :] = False
            elif dz == -1:
                mask[-1, :] = False
            if dm == 1:
                mask[:, 0] = False
            elif dm == -1:
                mask[:, -1] = False
            num += np.where(mask, g * ww, 0.0)
            den += np.where(mask, ww, 0.0)
    return np.where(den > 0, num / np.maximum(den, 1e-300), grid)
