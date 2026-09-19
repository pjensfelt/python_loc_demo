"""The simulated world: the true robot and the measurements it produces.

Kept apart from the filters so it is obvious which lines are simulating
reality and which are estimating it -- one of the questions the course README
asks the students.
"""

import numpy as np

from . import models
from .params import Params, DemoState


class World:
    """True robot pose plus landmark measurement generation."""

    def __init__(self, params: Params, rng=None):
        self.p = params
        self.rng = np.random.default_rng() if rng is None else rng
        self.reset()

    def reset(self):
        self.xt, self.yt, self.at = 0.0, 0.0, 0.0

    @property
    def pose(self):
        return self.xt, self.yt, self.at

    def step(self, state: DemoState):
        """Advance the true pose by one sampling interval.

        The true robot can itself be noisy: if the TRUE motion noise factors
        are non-zero the robot does not go exactly where it was commanded,
        which is what makes the true/model comparison interesting.  With them
        at zero it goes exactly where it was commanded.
        """
        D, DA = models.sample_motion_noise(
            state.tspeed, state.rspeed, self.p.dT,
            state.value("true", "td"),
            state.value("true", "rda"),
            state.value("true", "rd"),
            rng=self.rng,
        )
        self.xt, self.yt, self.at = models.motion_model(self.xt, self.yt, self.at, D, DA)

    def measure(self, state: DemoState):
        """Generate a noisy range and bearing to every landmark.

        All landmarks are always measured; `lmask` controls which measurements
        the *filter* is allowed to use, so turning a landmark off does not
        change the random stream of the others.
        """
        rho, phi = models.range_bearing(self.xt, self.yt, self.at, self.p.xL, self.p.yL)
        rho = rho + state.value("true", "rho") * self.rng.standard_normal(self.p.NL)
        phi = phi + state.value("true", "phi") * self.rng.standard_normal(self.p.NL)
        return rho, phi

    def disturb(self):
        """Teleport the true robot a little, to break the filter's tracking.

        The displacement is symmetric: +-0.25 m and +-30 degrees.
        """
        self.xt += self.rng.uniform(-0.25, 0.25)
        self.yt += self.rng.uniform(-0.25, 0.25)
        self.at += self.rng.uniform(-np.pi / 6, np.pi / 6)
