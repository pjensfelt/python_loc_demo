"""Monte Carlo Localization (particle filter).

A per-particle Python loop is far too slow at 10k-100k particles, so the
expressions below are vectorised over particles while keeping the explicit
loop over the (at most four) landmarks.  Each line still reads like the
scalar formula.

Weights are deliberately *not* normalised: they are multiplied by the
measurement likelihood at every update and only reset by resampling.  That is
what makes weight degeneracy visible when resampling is switched off.
"""

import numpy as np

from . import models
from .params import Params, DemoState


def resample_stratified(w, newN, rng=None):
    """Stratified resampling; returns `newN` indices into `w`.

    Jose-Luis Blanco's algorithm, written with searchsorted so it is
    O(N log N) instead of a Python loop.  Handles unnormalised weights and a
    target size different from len(w).
    """
    rng = np.random.default_rng() if rng is None else rng
    w = np.asarray(w, dtype=float)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        print("Weights are all zero, will not work to resample, resetting weights")
        w = np.ones_like(w) / len(w)
        total = w.sum()

    Q = np.cumsum(w)
    T = total * (rng.random(newN) + np.arange(newN)) / newN
    return np.searchsorted(Q, T, side="left").clip(0, len(w) - 1)


class ParticleFilter:
    def __init__(self, params: Params, N=100, rng=None):
        self.p = params
        self.rng = np.random.default_rng() if rng is None else rng
        self.reset(N)

    # ------------------------------------------------------------------
    @property
    def N(self):
        return self.X.shape[1]

    def reset(self, N=None):
        """All particles at the origin with equal weight."""
        N = self.N if N is None else N
        self.X = np.zeros((3, N))
        self.w = np.full(N, 1.0 / N)

    def set_uniform(self):
        """Spread the particles uniformly over the whole state space."""
        N = self.N
        self.X[0] = self.rng.uniform(*self.p.xlim, N)
        self.X[1] = self.rng.uniform(*self.p.ylim, N)
        self.X[2] = self.rng.uniform(0, 2 * np.pi, N)
        self.w = np.full(N, 1.0 / N)

    def set_size(self, newN):
        """Change the particle count by resampling to the new size.

        The new weights are uniform, as any resampling step should leave
        them -- carrying the parents' weights over to the children would
        silently reweight the set.
        """
        if newN == self.N:
            return
        idx = resample_stratified(self.w, newN, self.rng)
        self.X = self.X[:, idx]
        self.w = np.full(newN, 1.0 / newN)

    # ------------------------------------------------------------------
    def maybe_resample(self, state: DemoState):
        """Resample when the accumulated weight has decayed."""
        if state.resample and self.w.sum() < 0.5:
            idx = resample_stratified(self.w, self.N, self.rng)
            self.X = self.X[:, idx]
            self.w = np.full(self.N, 1.0 / self.N)
            return True
        return False

    def predict(self, state: DemoState):
        """Push every particle through the motion model with its own noise."""
        D, DA = models.sample_motion_noise(
            state.tspeed, state.rspeed, self.p.dT,
            state.value("model", "td"),
            state.value("model", "rda"),
            state.value("model", "rd"),
            size=self.N, rng=self.rng,
        )
        self.X[0], self.X[1], self.X[2] = models.motion_model(
            self.X[0], self.X[1], self.X[2], D, DA)

    def update(self, rho, phi, state: DemoState):
        """Multiply each particle's weight by p(z | x)."""
        for l in range(self.p.NL):
            if not state.lmask[l]:
                continue
            xl, yl = self.p.xL[l], self.p.yL[l]
            zRho, zPhi = models.range_bearing(self.X[0], self.X[1], self.X[2], xl, yl)

            if state.use_range:
                self.w *= np.exp(-0.5 * ((rho[l] - zRho) / state.zRhoStd) ** 2)

            if state.use_bearing:
                dPhi = models.wrap_angle(phi[l] - zPhi)
                self.w *= np.exp(-0.5 * (dPhi / state.zPhiStd) ** 2)

    def weights_are_tiny(self):
        return self.w.min() < 1e-200

    # ------------------------------------------------------------------
    def gaussian_approx(self, n_draw=1000):
        """Mean and xy covariance of the posterior, for the overlay.

        Estimated from a resampled (hence weight-free) subset.  Returns None
        when every draw comes from the same parent, since the covariance is
        then degenerate.
        """
        idx = resample_stratified(self.w, n_draw, self.rng)
        if idx.max() == idx.min():
            return None
        XX = self.X[:, idx]
        mu = XX[:2].mean(axis=1)
        sigma = np.cov(XX[:2])
        muA = np.arctan2(np.sin(XX[2]).mean(), np.cos(XX[2]).mean())
        return mu, sigma, muA
