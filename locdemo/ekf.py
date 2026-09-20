"""Extended Kalman filter for landmark based localization.

The whole filter is the two methods below.  Everything else in this package is
simulation, drawing or user interface.
"""

import numpy as np

from . import models
from .params import Params, DemoState


class EKFLocalizer:
    def __init__(self, params: Params):
        self.p = params
        self.X0 = np.zeros(3)
        self.P0 = 1e-6 * np.eye(3)
        self.reset()

    def reset(self):
        self.X = self.X0.copy()
        self.P = self.P0.copy()

    def set_uniform(self):
        """"Global localization" start: no idea where we are.

        A Gaussian cannot actually represent a uniform distribution, which is
        the point of the exercise -- watch how badly the EKF copes.
        """
        self.X = np.zeros(3)
        self.P = 1e4 * np.eye(3)

    def inject_noise(self, amount=0.1):
        self.P = self.P + amount * np.eye(3)

    # ------------------------------------------------------------------
    def predict(self, state: DemoState):
        """Propagate the estimate through the motion model.

        The commanded (v, w) is first passed through odometry_scale: with a
        wheel radius/wheelbase mismatch, D and DA are what the *odometry*
        believes happened, not what actually did -- a deterministic bias on
        top of the usual noise, and the reason a bad calibration here is so
        much worse than ordinary noise: it never averages out.
        """
        dT = self.p.dT
        v_scale, w_scale = models.odometry_scale(
            state.value("true", "r"), state.value("true", "B"), self.p.r, self.p.B)
        D, DA = state.tspeed * v_scale * dT, state.rspeed * w_scale * dT

        # Jacobians must be evaluated at the *previous* heading -- using the
        # already-propagated heading here would give the wrong linearisation.
        a_prev = self.X[2]
        A, W = models.motion_jacobians(a_prev, D)

        self.X[0], self.X[1], self.X[2] = models.motion_model(
            self.X[0], self.X[1], self.X[2], D, DA)

        # Process noise, expressed in the three error sources of the odometry
        # model: translation error from driving, rotation error from turning,
        # and rotation error from driving straight.
        Q = np.diag([(D * state.value("model", "td")) ** 2,
                     (DA * state.value("model", "rda")) ** 2,
                     (D * state.value("model", "rd")) ** 2])

        self.P = A @ self.P @ A.T + W @ Q @ W.T

    def update(self, rho, phi, state: DemoState):
        """Fuse the landmark measurements.

        All enabled measurements are stacked into one batch update, so H is
        (m x 3) with one row per scalar measurement.
        """
        H, R, innov = [], [], []
        x, y, a = self.X

        for l in range(self.p.NL):
            if not state.lmask[l]:
                continue
            xl, yl = self.p.xL[l], self.p.yL[l]
            zRho, zPhi = models.range_bearing(x, y, a, xl, yl)

            if state.use_range:
                H.append(models.range_jacobian(x, y, xl, yl, zRho))
                innov.append(rho[l] - zRho)
                R.append(state.zRhoStd ** 2)

            if state.use_bearing:
                H.append(models.bearing_jacobian(x, y, xl, yl, zRho))
                innov.append(models.wrap_angle(phi[l] - zPhi))
                R.append(state.zPhiStd ** 2)

        if not innov:
            return

        H = np.array(H)
        R = np.diag(R)
        innov = np.array(innov)

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.X = self.X + K @ innov
        self.P = (np.eye(3) - K @ H) @ self.P
