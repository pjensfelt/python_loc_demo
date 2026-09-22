"""Simulation parameters and mutable demo state.

The fixed world description lives in `Params`; everything the keyboard can
change lives in `DemoState`.

Every noise parameter exists in two versions:

    TRUE  -- used by the simulator to move the robot and generate measurements
    MODEL -- used by the filter (EKF covariances, particle spreading)

Being able to edit both columns independently lets you demonstrate
over-confident and under-confident filters directly: set MODEL smaller than
TRUE and watch the filter grow too sure of itself, or larger and watch it
throw away precision it could have had.

Two more rows, wheel radius `r` and wheelbase `B`, model a *deterministic*
error instead of noise: a differential-drive robot's odometry converts wheel
encoder ticks to distance using an assumed `r`/`B`, and if that assumption is
wrong the odometry is systematically biased even with perfect encoders. Real
hardware never machines a wheel to exactly the radius on its spec sheet, so
it's the *true* value that varies from one robot to the next -- the model's
assumed `r`/`B` is a fixed constant the controller was built around. These
two are therefore fixed on the MODEL side (`Params.r`/`Params.B` -- the
assumed calibration, not something to edit live) and editable only on the
TRUE side, via `odometry_scale` in `models.py`.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Params:
    """Fixed description of the world and the robot.  Not changed by the UI."""

    # Sampling rate [s]
    dT: float = 0.1

    # Robot footprint used for drawing [m]
    length: float = 0.3
    width: float = 0.2

    # Modelled (assumed) wheel radius and wheelbase [m], for the
    # odometry-bias demo. Fixed calibration constants the controller's
    # odometry assumes -- not editable live -- only the *true* hardware
    # values (the "r"/"B" true tunables below) can be detuned away from
    # these.
    r: float = 0.05
    B: float = 0.2

    # Modelled (assumed) compass bias [rad], same idea as r/B above: the
    # filter always assumes a perfectly calibrated compass; only the *true*
    # hardware value (the "compass_bias" true tunable) can be detuned away
    # from this.
    compass_bias: float = 0.0

    # Landmark positions
    xL: np.ndarray = field(default_factory=lambda: np.array([-0.5, 10.5, 10.5, -0.5]))
    yL: np.ndarray = field(default_factory=lambda: np.array([-0.5, -0.5, 10.5, 10.5]))

    # Axis limits, also used as the support of the uniform distribution
    xlim: tuple = (-1.0, 11.0)
    ylim: tuple = (-1.0, 11.0)

    @property
    def NL(self) -> int:
        return len(self.xL)

    def __post_init__(self):
        if len(self.xL) != len(self.yL):
            raise ValueError("xL and yL must have the same length")


# --------------------------------------------------------------------------
# Tunable parameters
# --------------------------------------------------------------------------
#
# Each tunable is edited by stepping along an explicit ladder of values rather
# than by dragging a slider.  Explicit numbers are easier to talk about during
# a lecture and make an experiment reproducible.

# Noise factors for the motion model.  The standard deviation of each error
# source is the factor times the distance (D) or the heading change (DA) of the
# step, so 0.0 means "no noise from this source". The top two rungs (2.0,
# 5.0) are deliberately far past anything a real robot would need: at that
# level the noise swamps the nominal step and the motion model stops
# looking directional at all, which is the point -- it's how to demo
# "prediction as if motion were random" live instead of only as a canned
# illustration. Appended at the *top* of the ladder rather than reached by
# stepping below 0.0: 0.0 is already the natural floor (no noise), and this
# ladder has no "off" state living below it the way the model rho/phi
# ladders do (see _RHO_MODEL/_PHI_MODEL below) -- reusing that convention
# here for an unrelated meaning (huge noise, not "ignore this") would
# invert the ladder's monotonic sense and contradict the one precedent that
# already exists for stepping past an end.
_MOTION = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.0, 5.0]

# Range [m].  The model column may be switched off entirely (-1); the true
# column cannot, but 0.0 gives a perfect sensor.
_RHO_TRUE = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
_RHO_MODEL = [-1.0] + _RHO_TRUE[1:]

# Bearing [rad], laddered in whole degrees.
_PHI_DEG = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
_PHI_TRUE = [np.deg2rad(d) for d in _PHI_DEG]
_PHI_MODEL = [-1.0] + _PHI_TRUE[1:]

# GPS/compass are fire-once sensors (see keys.py 'G'/'y'): whether one happens
# at all is already fully controlled by *pressing the key*, so unlike every
# continuous sensor above, there is no separate "off" state here -- it would
# just be a second, redundant way to suppress the same thing. Default is
# 5 m / 5 deg, in line with an uncorrected consumer GPS/compass rather than
# a survey-grade one.
#
# 0.0 is still offered on the TRUE side (an idealised perfect fix is a
# reasonable thing to demo), but dropped from MODEL: a zero model sigma is a
# divide-by-zero in the likelihood/Kalman gain the moment the fix doesn't
# land exactly on the prediction, same reason rho/phi's model ladder never
# actually reaches their own 0.0 either (it's replaced by "off" there).
_GPS_TRUE = [0.0, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0]
_GPS_MODEL = [v for v in _GPS_TRUE if v > 0]

_COMPASS_DEG = [0.0, 0.1, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
_COMPASS_TRUE = [np.deg2rad(d) for d in _COMPASS_DEG]
_COMPASS_MODEL = [np.deg2rad(d) for d in _COMPASS_DEG if d > 0]

# Wheel radius and wheelbase [m], laddered around the modelled values (0.05,
# 0.2) declared on Params -- there is no "off" here, since the real hardware
# always has *some* r/B, even when it happens to match the model exactly.
#
# r: 1mm resolution throughout, capped at +/-2cm -- being off by more than
# that on a 5cm wheel seems unlikely in practice.
_R_MODEL = 0.05  # must match Params.r
_R_DEV_MM = list(range(-20, 21))                    # +/-20mm in 1mm steps
_R_LADDER = [round(_R_MODEL + d / 1000, 6) for d in _R_DEV_MM]
_R_DEFAULT = _R_DEV_MM.index(0)

# B: 1mm resolution close to the modelled value (+/-1cm), coarser 1cm steps
# further out, capped at +/-10cm total.
_B_MODEL = 0.2  # must match Params.B
_B_MAG_MM = sorted(set(range(0, 11)) | set(range(20, 101, 10)))  # 0..10, 20..100
_B_DEV_MM = sorted({-m for m in _B_MAG_MM} | set(_B_MAG_MM))
_B_LADDER = [round(_B_MODEL + d / 1000, 6) for d in _B_DEV_MM]
_B_DEFAULT = _B_DEV_MM.index(0)

# Compass bias [rad], laddered in whole degrees, +/-30 deg -- a fixed,
# deterministic offset (hard-iron-style), the same *kind* of error as the
# wheel r/B above: true-only, no "off", since real hardware always has
# *some* bias even when it happens to be zero.
_COMPASS_BIAS_MODEL = 0.0  # must match Params.compass_bias
_COMPASS_BIAS_DEV_DEG = list(range(-30, 31))
_COMPASS_BIAS_LADDER = [np.deg2rad(_COMPASS_BIAS_MODEL + d) for d in _COMPASS_BIAS_DEV_DEG]
_COMPASS_BIAS_DEFAULT = _COMPASS_BIAS_DEV_DEG.index(0)


@dataclass(frozen=True)
class Tunable:
    """One editable quantity, in its true or its modelled version."""

    name: str        # 'td', 'rda', 'rd', 'rho', 'phi'
    column: str      # 'true' or 'model'
    label: str       # shown in the parameter panel
    ladder: List[float]
    default: int     # index into ladder
    unit: str = ""   # '', 'm' or 'deg'

    @property
    def key(self):
        return (self.column, self.name)

    def format(self, value: float) -> str:
        if value < 0:
            return "off"
        if self.unit == "deg":
            return f"{np.rad2deg(value):.3g}°"
        if self.unit == "m":
            return f"{value:.3g}m"
        return f"{value:.3g}"


# Display order is also the order the selection cursor walks through.
TUNABLES: List[Tunable] = [
    # motion noise, proportional to distance travelled
    Tunable("td", "true", "sig_td", _MOTION, _MOTION.index(0.0)),
    Tunable("td", "model", "sig_td", _MOTION, _MOTION.index(0.1)),
    # rotation noise, proportional to the heading change
    Tunable("rda", "true", "sig_rda", _MOTION, _MOTION.index(0.0)),
    Tunable("rda", "model", "sig_rda", _MOTION, _MOTION.index(0.1)),
    # rotation noise, proportional to distance travelled
    Tunable("rd", "true", "sig_rd", _MOTION, _MOTION.index(0.0)),
    Tunable("rd", "model", "sig_rd", _MOTION, _MOTION.index(0.1)),
    # range
    Tunable("rho", "true", "sig_rho", _RHO_TRUE, _RHO_TRUE.index(0.1), "m"),
    Tunable("rho", "model", "sig_rho", _RHO_MODEL, 0, "m"),
    # bearing
    Tunable("phi", "true", "sig_phi", _PHI_TRUE, _PHI_DEG.index(1.0), "deg"),
    Tunable("phi", "model", "sig_phi", _PHI_MODEL, 0, "deg"),
    # GPS: one-shot absolute position fix -- no "off", see _GPS_TRUE above
    # Default 1m, not the 5m a real uncorrected consumer GPS would have:
    # the world here is only ~12m across (see Params.xlim/ylim), so a 5m
    # std routinely lands a reading several metres from the true robot --
    # correct given that sigma, but it swamps the whole map and makes a
    # single fix look broken by default. 5/10 stay on the ladder for
    # demoing a deliberately bad GPS.
    Tunable("gps", "true", "sig_gps", _GPS_TRUE, _GPS_TRUE.index(1.0), "m"),
    Tunable("gps", "model", "sig_gps", _GPS_MODEL, _GPS_MODEL.index(1.0), "m"),
    # compass: one-shot absolute heading fix -- no "off", see _COMPASS_DEG above
    Tunable("compass", "true", "sig_cmp", _COMPASS_TRUE, _COMPASS_DEG.index(5.0), "deg"),
    Tunable("compass", "model", "sig_cmp", _COMPASS_MODEL,
            _COMPASS_MODEL.index(np.deg2rad(5.0)), "deg"),
    # wheel radius, wheelbase and compass bias -- true-only, see
    # FIXED_MODEL_ROWS below
    Tunable("r", "true", "r", _R_LADDER, _R_DEFAULT, "m"),
    Tunable("B", "true", "B", _B_LADDER, _B_DEFAULT, "m"),
    Tunable("compass_bias", "true", "cmp_bias", _COMPASS_BIAS_LADDER, _COMPASS_BIAS_DEFAULT, "deg"),
]

TUNABLE_BY_KEY = {t.key: t for t in TUNABLES}
PARAM_ROWS = ["td", "rda", "rd", "rho", "phi"]

# GPS/compass: true+model rows like PARAM_ROWS, but kept separate and drawn
# under their own divider, since they have no "off" state -- see the
# _GPS_TRUE/_COMPASS_DEG comment above for why that would be redundant here.
FIRE_ONCE_ROWS = ["gps", "compass"]

# Rows with no "model" Tunable at all: the model value is a fixed constant
# on Params (r/B/compass_bias), not something edited via the ladder
# mechanism. Kept separate from PARAM_ROWS since callers that need the model
# value for one of these must read it off Params instead of calling
# state.value("model", ...).
FIXED_MODEL_ROWS = ["r", "B", "compass_bias"]

# Particle counts offered by the 'n' / 'N' keys (was a popup menu).  The
# filter math itself is vectorised and comfortably handles far more than
# this top rung; what doesn't scale is drawing them, which is why
# ParticleArtist only ever draws a fixed-size random subset (see draw.py) no
# matter how big N gets -- or all of them, slowly, with 'A'.
N_LADDER = [100, 1000, 10000, 100000]

# Speed limits and step sizes for the arrow keys
V_STEP, V_MAX = 0.05, 1.0
W_STEP, W_MAX = np.deg2rad(5.0), np.deg2rad(115.0)


@dataclass
class DemoState:
    """Everything the keyboard can change, plus one-shot request flags.

    The main loop polls the one-shot flags each step and clears them once
    handled.
    """

    running: bool = True
    tspeed: float = 0.0
    rspeed: float = 0.0

    # One-shot requests
    reset: bool = False
    setUniform: bool = False
    addDisturbance: bool = False
    forceUpdate: bool = False
    injectNoise: bool = False
    superGPS: bool = False
    resampleOnce: bool = False
    injectGPS: bool = False
    injectCompass: bool = False

    # Which landmarks are in use
    lmask: np.ndarray = field(default_factory=lambda: np.ones(4, dtype=bool))

    # Display / filter options
    dispGaussApprox: bool = False
    # Particle colouring, one of "plain"/"weight"/"heading" -- a single mode
    # rather than two independent toggles, since "coloured by weight" and
    # "coloured by heading" can't both be true on screen at once anyway; two
    # booleans just meant one silently overrode the other with no feedback
    # about why the other key seemed to do nothing.
    ptColorMode: str = "weight"
    drawAllParticles: bool = False
    resample: bool = False
    showTrueRobot: bool = True

    # Particle count
    n_idx: int = 0
    newN: int = 100

    # Indices into each tunable's ladder, plus the selection cursor
    idx: dict = field(default_factory=lambda: {t.key: t.default for t in TUNABLES})
    cursor: int = 0

    # Saved (rho, phi) model indices while the 'x' key has forced them to
    # "off"; None means the exteroceptive model noise is at its normal value.
    _extero_saved: Optional[Tuple[int, int]] = None

    # ---------------- tunable access ----------------

    def value(self, column: str, name: str) -> float:
        t = TUNABLE_BY_KEY[(column, name)]
        return t.ladder[self.idx[t.key]]

    def step(self, column: str, name: str, delta: int) -> None:
        t = TUNABLE_BY_KEY[(column, name)]
        self.idx[t.key] = int(np.clip(self.idx[t.key] + delta, 0, len(t.ladder) - 1))

    def set_value(self, column: str, name: str, value: float) -> None:
        """Snap a tunable to the ladder entry closest to `value`."""
        t = TUNABLE_BY_KEY[(column, name)]
        self.idx[t.key] = int(np.argmin([abs(v - value) for v in t.ladder]))

    @property
    def selected(self) -> Tunable:
        return TUNABLES[self.cursor]

    def link_selected(self, params: Params) -> None:
        """Zero out the bias on the selected row.

        For an ordinary row that's model := true. For a FIXED_MODEL_ROWS row
        (wheel r/B) the model is the fixed constant on Params, so it's the
        editable true side that gets reset to match it instead.
        """
        name = self.selected.name
        if name in FIXED_MODEL_ROWS:
            self.set_value("true", name, getattr(params, name))
        else:
            self.set_value("model", name, self.value("true", name))

    def link_all(self, params: Params) -> None:
        for name in PARAM_ROWS + FIRE_ONCE_ROWS:
            self.set_value("model", name, self.value("true", name))
        for name in FIXED_MODEL_ROWS:
            self.set_value("true", name, getattr(params, name))

    def toggle_extero(self) -> None:
        """Force the modelled range/bearing noise to 'off', or restore it.

        A quick way to switch the filter to dead reckoning and back without
        losing the sig_rho/sig_phi values you had dialled in.
        """
        if self._extero_saved is None:
            self._extero_saved = (self.idx[("model", "rho")], self.idx[("model", "phi")])
            self.idx[("model", "rho")] = 0
            self.idx[("model", "phi")] = 0
        else:
            self.idx[("model", "rho")], self.idx[("model", "phi")] = self._extero_saved
            self._extero_saved = None

    @property
    def extero_off(self) -> bool:
        return self._extero_saved is not None

    # ---------------- convenience shorthands ----------------

    @property
    def zRhoStd(self) -> float:
        return self.value("model", "rho")

    @property
    def zPhiStd(self) -> float:
        return self.value("model", "phi")

    @property
    def use_range(self) -> bool:
        return self.zRhoStd > 0

    @property
    def use_bearing(self) -> bool:
        return self.zPhiStd > 0

    @property
    def zGpsStd(self) -> float:
        return self.value("model", "gps")

    @property
    def zCompassStd(self) -> float:
        return self.value("model", "compass")

    @property
    def moving(self) -> bool:
        return abs(self.tspeed) > 1e-6 or abs(self.rspeed) > 1e-6
