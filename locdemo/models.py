"""Motion and measurement models, shared by the simulator and both filters.

Every function here works both on scalars and on arrays of particles, so the
particle filter can evaluate a whole sample set with the same expressions the
EKF uses for a single state.
"""

import numpy as np


def wrap_angle(a):
    """Wrap an angle (or array of angles) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def motion_model(x, y, a, D, DA):
    """Move a pose by a distance D along its heading and then turn by DA.

    D and DA may be arrays, which is how the particle filter injects a
    different noise realisation into every particle.
    """
    return x + D * np.cos(a), y + D * np.sin(a), a + DA


def sample_motion_noise(v, w, dT, td_std, rda_std, rd_std, size=None, rng=None):
    """Draw the three motion error sources of the odometry model.

    Returns (D, DA): the noisy distance travelled and heading change.  The
    standard deviation of each source is proportional to the nominal distance
    D = v*dT or heading change DA = w*dT, so a stationary robot accumulates no
    error.  The particle filter draws from this directly; the EKF uses its
    linearised form instead.
    """
    rng = np.random.default_rng() if rng is None else rng
    D, DA = v * dT, w * dT
    dnoise = (D * td_std) * rng.standard_normal(size)    # distance error
    arnoise = (DA * rda_std) * rng.standard_normal(size)  # turn error from turning
    atnoise = (D * rd_std) * rng.standard_normal(size)    # turn error from driving
    return D + dnoise, DA + arnoise + atnoise


def range_bearing(x, y, a, xl, yl):
    """Predicted range and bearing from pose (x, y, a) to landmark (xl, yl)."""
    dx, dy = xl - x, yl - y
    return np.hypot(dx, dy), np.arctan2(dy, dx) - a


def motion_jacobians(a, D):
    """Jacobians of motion_model at heading `a` over a step of length D.

    A -- d f / d state, W -- d f / d noise, with the noise vector ordered as
    (distance error, turn-from-turning error, turn-from-driving error).

    NOTE: `a` must be the heading *before* the step; evaluating it at the
    already-propagated heading instead is a small but real error at dT = 0.1.
    """
    A = np.array([[1.0, 0.0, -D * np.sin(a)],
                  [0.0, 1.0, D * np.cos(a)],
                  [0.0, 0.0, 1.0]])
    W = np.array([[np.cos(a), 0.0, 0.0],
                  [np.sin(a), 0.0, 0.0],
                  [0.0, 1.0, 1.0]])
    return A, W


def range_jacobian(x, y, xl, yl, rho):
    """Row of H for a range measurement to one landmark."""
    return np.array([-(xl - x) / rho, -(yl - y) / rho, 0.0])


def bearing_jacobian(x, y, xl, yl, rho):
    """Row of H for a bearing measurement to one landmark."""
    return np.array([(yl - y) / rho ** 2, -(xl - x) / rho ** 2, -1.0])
