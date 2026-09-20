"""EKF-SLAM: like EKFLocalizer, but the landmark positions are unknown and
join the state vector the first time each one is seen.

State layout: [x, y, a, lx_1, ly_1, lx_2, ly_2, ...], in the order landmarks
were first observed. Robot prediction and the range/bearing measurement model
are exactly the ones in `models.py`; only the bookkeeping for a growing state
is new.
"""

import numpy as np

from . import models
from .params import Params, DemoState

# Initial variance given to a freshly mapped landmark: uncorrelated with the
# rest of the state and deliberately huge, i.e. "we have no idea". The very
# first fused update after that already correlates the landmark with the
# robot pose -- H references both blocks at once, and (I - K H) mixes them
# into P regardless of how small that update's innovation happens to be. What
# repeated sightings of the *same* landmark then do is shrink its variance
# down from this huge starting value (and sharpen that correlation) towards
# something the eye can actually see as a tilted ellipse -- which is what the
# "landmark inherits the robot's uncertainty" demo is showing.
INIT_LANDMARK_VAR = 1e10


class EKFSLAM:
    def __init__(self, params: Params):
        self.p = params
        self.X0 = np.zeros(3)
        self.P0 = 1e-6 * np.eye(3)
        self.reset()

    def reset(self):
        self.X = self.X0.copy()
        self.P = self.P0.copy()
        self.landmark_index = {}  # landmark id -> index of its x in self.X

    def set_uniform(self):
        """"Global localization" for the robot pose only.

        Severs the correlation with any already-mapped landmarks too: if we
        no longer know where we are, whatever the map told us about how it
        connects to us is void as well.
        """
        self.X[:3] = 0.0
        self.P[:3, :3] = 1e4 * np.eye(3)
        self.P[:3, 3:] = 0.0
        self.P[3:, :3] = 0.0

    def inject_noise(self, amount=0.1):
        self.P[:3, :3] += amount * np.eye(3)

    def mapped_landmarks(self):
        """(landmark id, mean, covariance) for every currently mapped landmark."""
        return [(l, self.X[i:i + 2].copy(), self.P[i:i + 2, i:i + 2].copy())
                for l, i in sorted(self.landmark_index.items())]

    # ------------------------------------------------------------------
    def predict(self, state: DemoState):
        """Propagate robot pose and covariance; mapped landmarks don't move.

        A and W are full-state versions of the 3x3 motion Jacobians, padded
        with an identity block for the landmarks, so the same
        `A @ P @ A.T + W @ Q @ W.T` line as EKFLocalizer also carries the
        robot-landmark cross-covariance blocks through correctly.
        """
        dT = self.p.dT
        v_scale, w_scale = models.odometry_scale(
            state.value("true", "r"), state.value("true", "B"), self.p.r, self.p.B)
        D, DA = state.tspeed * v_scale * dT, state.rspeed * w_scale * dT
        N = len(self.X)

        a_prev = self.X[2]
        A3, W3 = models.motion_jacobians(a_prev, D)

        A = np.eye(N)
        A[:3, :3] = A3
        W = np.zeros((N, 3))
        W[:3, :] = W3

        self.X[0], self.X[1], self.X[2] = models.motion_model(
            self.X[0], self.X[1], self.X[2], D, DA)

        Q = np.diag([(D * state.value("model", "td")) ** 2,
                     (DA * state.value("model", "rda")) ** 2,
                     (D * state.value("model", "rd")) ** 2])

        # errstate: mixing a freshly-mapped landmark's ~1e10 variance with
        # everything else's much smaller one triggers spurious divide/overflow
        # FP flags in Apple's Accelerate BLAS, even though the result itself
        # is finite -- see the same guard in _apply_update.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            self.P = A @ self.P @ A.T + W @ Q @ W.T

    def update(self, rho, phi, state: DemoState):
        """Map any landmark seen for the first time, then fuse measurements.

        All enabled measurements of already-mapped landmarks are stacked
        into one batch update, exactly as in EKFLocalizer.update. A
        landmark's own column pair gets minus the robot's (x, y) partials,
        since range and bearing depend only on the relative position
        landmark-minus-robot.

        With both measurement types off (model.rho and model.phi are both
        "off") there is no way to ever fuse a new landmark's readings, so it
        would just sit in the state forever at its huge initial variance --
        this returns before mapping anything in that case, rather than
        growing the state for no benefit.
        """
        if not (state.use_range or state.use_bearing):
            return

        x, y, a = self.X[0], self.X[1], self.X[2]
        for l in range(self.p.NL):
            if not state.lmask[l] or l in self.landmark_index:
                continue
            self.landmark_index[l] = len(self.X)
            self.X = np.concatenate([self.X, [
                x + rho[l] * np.cos(a + phi[l]),
                y + rho[l] * np.sin(a + phi[l]),
            ]])
            n = len(self.P)
            self.P = np.block([
                [self.P, np.zeros((n, 2))],
                [np.zeros((2, n)), INIT_LANDMARK_VAR * np.eye(2)],
            ])

        H, R, innov = [], [], []
        N = len(self.X)
        x, y, a = self.X[0], self.X[1], self.X[2]

        for l in range(self.p.NL):
            if not state.lmask[l]:
                continue
            xpos = self.landmark_index[l]
            xl, yl = self.X[xpos], self.X[xpos + 1]
            zRho, zPhi = models.range_bearing(x, y, a, xl, yl)

            if state.use_range:
                row = np.zeros(N)
                row[:3] = models.range_jacobian(x, y, xl, yl, zRho)
                row[xpos], row[xpos + 1] = -row[0], -row[1]
                H.append(row)
                innov.append(rho[l] - zRho)
                R.append(state.zRhoStd ** 2)

            if state.use_bearing:
                row = np.zeros(N)
                row[:3] = models.bearing_jacobian(x, y, xl, yl, zRho)
                row[xpos], row[xpos + 1] = -row[0], -row[1]
                H.append(row)
                innov.append(models.wrap_angle(phi[l] - zPhi))
                R.append(state.zPhiStd ** 2)

        if not innov:
            return
        self._apply_update(np.array(H), np.diag(R), np.array(innov))

    def super_gps_update(self, xt, yt, var=1e-6):
        """A near-perfect direct fix of the robot's true (x, y).

        Unlike a real sensor this reads the ground truth `xt, yt` straight
        from the world, with only a tiny nominal variance to keep the
        update well-conditioned. It exists to show how one precise fix drags
        every *correlated* landmark along with the robot, not to model an
        actual sensor.
        """
        N = len(self.X)
        H = np.zeros((2, N))
        H[0, 0] = H[1, 1] = 1.0
        innov = np.array([xt - self.X[0], yt - self.X[1]])
        self._apply_update(H, var * np.eye(2), innov)

    def _apply_update(self, H, R, innov):
        # See the errstate note in predict(): same spurious-warning cause,
        # here from multiplying through a landmark's huge initial variance.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S)
            self.X = self.X + K @ innov
            self.P = (np.eye(len(self.X)) - K @ H) @ self.P
