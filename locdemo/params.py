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
wrong the odometry is systematically biased even with perfect encoders. These
two are fixed on the TRUE side (`Params.r`/`Params.B` -- real hardware, not
something to edit live) and editable only on the MODEL side, via
`odometry_scale` in `models.py`.
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

    # True wheel radius and wheelbase [m], for the odometry-bias demo.
    # Fixed hardware, not editable live -- only the odometry's *belief*
    # about them (the "r"/"B" model tunables below) can be detuned.
    r: float = 0.05
    B: float = 0.2

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
# step, so 0.0 means "no noise from this source".
_MOTION = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0]

# Range [m].  The model column may be switched off entirely (-1); the true
# column cannot, but 0.0 gives a perfect sensor.
_RHO_TRUE = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
_RHO_MODEL = [-1.0] + _RHO_TRUE[1:]

# Bearing [rad], laddered in whole degrees.
_PHI_DEG = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
_PHI_TRUE = [np.deg2rad(d) for d in _PHI_DEG]
_PHI_MODEL = [-1.0] + _PHI_TRUE[1:]

# Wheel radius and wheelbase [m], laddered around the true values (0.05,
# 0.2) declared on Params -- there is no "off" here, since odometry always
# assumes *some* r/B, even when it happens to be the right one.
#
# r: 1mm resolution throughout, capped at +/-2cm -- being off by more than
# that on a 5cm wheel seems unlikely in practice.
_R_TRUE = 0.05  # must match Params.r
_R_DEV_MM = list(range(-20, 21))                    # +/-20mm in 1mm steps
_R_LADDER = [round(_R_TRUE + d / 1000, 6) for d in _R_DEV_MM]
_R_DEFAULT = _R_DEV_MM.index(0)

# B: 1mm resolution close to the true value (+/-1cm), coarser 1cm steps
# further out, capped at +/-10cm total.
_B_TRUE = 0.2  # must match Params.B
_B_MAG_MM = sorted(set(range(0, 11)) | set(range(20, 101, 10)))  # 0..10, 20..100
_B_DEV_MM = sorted({-m for m in _B_MAG_MM} | set(_B_MAG_MM))
_B_LADDER = [round(_B_TRUE + d / 1000, 6) for d in _B_DEV_MM]
_B_DEFAULT = _B_DEV_MM.index(0)


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
    # wheel radius and wheelbase -- model-only, see FIXED_TRUE_ROWS below
    Tunable("r", "model", "r", _R_LADDER, _R_DEFAULT, "m"),
    Tunable("B", "model", "B", _B_LADDER, _B_DEFAULT, "m"),
]

TUNABLE_BY_KEY = {t.key: t for t in TUNABLES}
PARAM_ROWS = ["td", "rda", "rd", "rho", "phi"]

# Rows with no "true" Tunable at all: the true value is the fixed
# Params.r/Params.B, not something edited via the ladder mechanism. Kept
# separate from PARAM_ROWS since callers that need the true value for one of
# these must read it off Params instead of calling state.value("true", ...).
FIXED_TRUE_ROWS = ["r", "B"]

# Particle counts offered by the 'n' / 'N' keys (was a popup menu).  The
# filter math is vectorised and handles a million particles in ~50ms; what
# doesn't scale is drawing them, which is why ParticleArtist only ever draws
# a fixed-size random subset (see draw.py) no matter how big N gets.
N_LADDER = [100, 1000, 10000, 100000, 1000000]

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

    # Which landmarks are in use
    lmask: np.ndarray = field(default_factory=lambda: np.ones(4, dtype=bool))

    # Display / filter options
    dispGaussApprox: bool = True
    coloredPts: bool = True
    resample: bool = False

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

    def true_value(self, name: str, params: Params) -> float:
        """The true value of a row, whether it's a ladder Tunable or one of
        FIXED_TRUE_ROWS living on Params instead."""
        if name in FIXED_TRUE_ROWS:
            return getattr(params, name)
        return self.value("true", name)

    def link_selected(self, params: Params) -> None:
        """Copy the true value of the selected row into its model entry."""
        name = self.selected.name
        self.set_value("model", name, self.true_value(name, params))

    def link_all(self, params: Params) -> None:
        for name in PARAM_ROWS + FIXED_TRUE_ROWS:
            self.set_value("model", name, self.true_value(name, params))

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
    def moving(self) -> bool:
        return abs(self.tspeed) > 1e-6 or abs(self.rspeed) > 1e-6
