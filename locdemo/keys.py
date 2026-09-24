"""Keyboard control.

Driving is set-point control -- you set v and w and the robot keeps going --
so the arrow keys nudge v and w up and down rather than driving the robot only
while held.  Space stops.
"""

import time

import numpy as np
import matplotlib as mpl

from . import app
from .params import DemoState, N_LADDER, V_STEP, V_MAX, W_STEP, W_MAX, TUNABLES, row_is_relevant

# Below this gap between two presses of the *same* key, the second one is
# dropped. A human tapping a key deliberately is never this fast, so this
# only ever suppresses OS/window-manager key auto-repeat -- or, worse, a
# stuck key event that keeps re-firing after you've let go, which otherwise
# looks exactly like the filter "updating on its own". This alone is not
# enough on its own for a key held down *longer* than the gap, though: OS
# auto-repeat still slips one press through roughly every _MIN_REPEAT_INTERVAL
# for as long as the key stays down. That's fine for driving (holding an
# arrow key is supposed to keep nudging v/w), but for a one-shot action --
# especially 'G'/'y', whose effect on the particle weights *compounds* with
# every repeat -- an accidental hold of a fraction of a second silently
# fires the fix several times over, each one further decaying sum(w), with
# no visible cause since the robot never moved. _CONTINUOUS_KEYS are exempt
# from the stricter check below and keep the old repeat-while-held
# behaviour; every other key only fires once per physical press, confirmed
# by tracking actual key-release events rather than just a time gap.
_MIN_REPEAT_INTERVAL = 0.15
_CONTINUOUS_KEYS = {"up", "down", "left", "right"}

# One key cycles both particle-colouring modes, rather than two independent
# toggles -- "coloured by weight" and "coloured by heading" can never both be
# true on screen at once, so two separate keys just meant pressing one
# silently did nothing while the other mode was active, with no indication
# why.
_PT_COLOR_MODES = ("weight", "heading")

_COMMON_HELP = """
 driving            filter / display        parameters
 -------            ----------------        ----------
 up/down    v +-    r  reset                tab   select next parameter
 left/right w +-    u  uniform              S-tab select previous
 space      stop    d  disturb true pose    >     increase selected
 0          w = 0   enter force an update   <     decrease selected
 1..4  toggle       g  Gaussian overlay     l     zero bias (this row)
       landmark     x  extero. noise on/off L     zero bias (all rows)
 h     this help    t  true robot on/off
 q     quit         S  screenshot (2 pngs)
                    H  set home (r returns here)
                    R  clear home (back to 0,0,0)"""

_EKF_ONLY = """
                    i  inject noise (EKF/SLAM)"""

_PF_ONLY = """
                    c  cycle colour mode    n/N   fewer/more particles
                       (weight/heading)     p     resampling on/off
                    A  draw all particles   o     resample once
                       (slow at high N)"""

_SLAM_ONLY = """
                    s  superGPS fix (SLAM)"""

_ABSOLUTE_ONLY = """
                    G  GPS fix (once)         y  compass fix (once)"""

_PGO_ONLY = """
                    O  optimize the graph (once)
                    G  GPS fix -- adds an edge, folded in on next O"""


def help_text(particles: bool = False, slam: bool = False, absolute: bool = False,
              pgo: bool = False) -> str:
    """The key list for the current demo; PF, EKF, SLAM and PGO each have a
    few keys the others don't, so they get their own copy of the mode-
    specific line. SLAM shares 'i' with EKF (both are Kalman filters) and
    adds 's'. `absolute` adds the GPS/compass one-shot fixes (EKF and PF,
    not SLAM, which already has its own near-perfect 's' fix for a
    different purpose). `pgo` is its own demo (run_pgo.py), neither PF-like
    nor EKF-like, so it gets no 'i'/'s' line at all, just its own key.
    """
    if pgo:
        extra = _PGO_ONLY
    elif particles:
        extra = _PF_ONLY
    else:
        extra = _EKF_ONLY + (_SLAM_ONLY if slam else "")
    extra += _ABSOLUTE_ONLY if absolute else ""
    return _COMMON_HELP + extra + "\n"


# Kept for backwards compatibility with anything importing the old constant;
# prefer help_text() so the mode-specific lines are correct.
HELP = help_text(particles=False)


def clear_default_keymap():
    """Stop matplotlib's own shortcuts from stealing our keys.

    Without this, 's' saves a PNG, 'q' closes the window, 'g' toggles the grid
    and 'l' switches the y axis to log scale.
    """
    for k in list(mpl.rcParams):
        if k.startswith("keymap."):
            mpl.rcParams[k] = []


def make_handler(state: DemoState, params, fig=None, ax=None, demo="demo", on_help=None,
                  particles=False, slam=False, absolute=False, pgo=False):
    """Return a matplotlib key_press_event callback bound to `state`.

    `params` is needed for 'l'/'L': the model value of a FIXED_MODEL_ROWS
    tunable (wheel r/B) lives on Params, not on a ladder in `state`.

    `ax` is only needed for 'S' (screenshot); passing None just disables
    that key instead of erroring, so callers that don't have an axes yet
    (headless runs) don't need to special-case anything. `demo` names the
    calling program (e.g. "pf", "ekf") for the screenshot filename.

    `particles` selects which of the PF-only / EKF-only keys are live,
    `slam` additionally enables the superGPS key, `absolute` enables the
    GPS/compass one-shot fixes -- EKF and PF, not SLAM, which already has
    its own near-perfect 's' fix serving a different demo -- and `pgo`
    enables the graph-optimization key (run_pgo.py only).
    """

    def clamp(value, limit):
        return float(np.clip(value, -limit, limit))

    # tab/shift-tab must only visit rows Panel.update() actually draws --
    # otherwise it's easy to land the cursor on a row that's blanked out for
    # this demo (see row_is_relevant), with no brackets anywhere to show
    # what's selected. Computed once, not per keypress: relevance only
    # depends on the flags this handler was built with, never on live state.
    visible = [i for i, t in enumerate(TUNABLES)
              if row_is_relevant(t.name, absolute=absolute, slam=slam, pgo=pgo)]

    last_press = {}
    held = set()

    def release(event):
        held.discard(event.key)

    def handler(event):
        k = event.key
        if k is None:
            return

        if k not in _CONTINUOUS_KEYS:
            if k in held:
                # Still down from an earlier press -- this is OS auto-repeat,
                # not a new physical press, so it doesn't fire again no
                # matter how long the key stays held.
                return
            held.add(k)

        now = time.monotonic()
        if now - last_press.get(k, -1.0) < _MIN_REPEAT_INTERVAL:
            return
        last_press[k] = now

        # ---- driving -------------------------------------------------
        if k == "up":
            state.tspeed = clamp(state.tspeed + V_STEP, V_MAX)
        elif k == "down":
            state.tspeed = clamp(state.tspeed - V_STEP, V_MAX)
        elif k == "left":
            state.rspeed = clamp(state.rspeed + W_STEP, W_MAX)
        elif k == "right":
            state.rspeed = clamp(state.rspeed - W_STEP, W_MAX)
        elif k == " ":
            state.tspeed = state.rspeed = 0.0
        elif k == "0":
            state.rspeed = 0.0

        # ---- one-shot requests --------------------------------------
        elif k == "r":
            state.reset = True
            state.tspeed = state.rspeed = 0.0
        elif k == "u":
            state.setUniform = True
        elif k == "d":
            state.addDisturbance = True
        elif k == "H":
            state.setHome = True
        elif k == "R":
            state.clearHome = True
        elif k == "i" and not particles and not pgo:
            state.injectNoise = True
        elif k == "enter":
            state.forceUpdate = True
        elif k == "s" and slam:
            state.superGPS = True
        elif k == "o" and particles:
            state.resampleOnce = True
        elif k == "G" and (absolute or pgo):
            state.injectGPS = True
        elif k == "y" and absolute:
            state.injectCompass = True
        elif k == "O" and pgo:
            state.optimizePGO = True

        # ---- toggles -------------------------------------------------
        elif k == "g":
            state.dispGaussApprox = not state.dispGaussApprox
        elif k == "x":
            state.toggle_extero()
        elif k == "t":
            state.showTrueRobot = not state.showTrueRobot
        elif k == "c" and particles:
            i = (_PT_COLOR_MODES.index(state.ptColorMode) + 1) % len(_PT_COLOR_MODES)
            state.ptColorMode = _PT_COLOR_MODES[i]
        elif k == "A" and particles:
            state.drawAllParticles = not state.drawAllParticles
        elif k == "p" and particles:
            state.resample = not state.resample
        elif k in "1234":
            i = int(k) - 1
            if i < len(state.lmask):
                state.lmask[i] = not state.lmask[i]

        # ---- parameter editing ---------------------------------------
        elif k == "tab":
            pos = visible.index(state.cursor) if state.cursor in visible else -1
            state.cursor = visible[(pos + 1) % len(visible)]
        elif "tab" in k.lower():
            # Shift-Tab's key string is backend- and platform-dependent --
            # "shift+tab" on some, the bare special name "backtab" on
            # others (e.g. macOS's native backend) -- so anything
            # Tab-flavoured that isn't plain "tab" is treated as the
            # backward step, rather than guessing at exact literals.
            pos = visible.index(state.cursor) if state.cursor in visible else 0
            state.cursor = visible[(pos - 1) % len(visible)]
        elif k == ">":
            t = state.selected
            state.step(t.column, t.name, +1)
        elif k == "<":
            t = state.selected
            state.step(t.column, t.name, -1)
        elif k == "l":
            state.link_selected(params)
        elif k == "L":
            state.link_all(params)

        # ---- particle count ------------------------------------------
        elif k == "N" and particles:
            state.n_idx = min(state.n_idx + 1, len(N_LADDER) - 1)
            state.newN = N_LADDER[state.n_idx]
        elif k == "n" and particles:
            state.n_idx = max(state.n_idx - 1, 0)
            state.newN = N_LADDER[state.n_idx]

        # ---- meta -----------------------------------------------------
        elif k == "h":
            print(help_text(particles, slam, absolute, pgo))
            if on_help is not None:
                on_help()
        elif k == "S" and ax is not None:
            app.save_screenshot(fig, ax, demo)
        elif k == "q":
            state.running = False
            if fig is not None:
                import matplotlib.pyplot as plt
                plt.close(fig)

    return handler, release


def connect(fig, state: DemoState, params, ax=None, demo="demo", on_help=None,
            particles=False, slam=False, absolute=False, pgo=False):
    clear_default_keymap()
    handler, release = make_handler(state, params, fig, ax, demo, on_help,
                                     particles, slam, absolute, pgo)
    fig.canvas.mpl_connect("key_press_event", handler)
    fig.canvas.mpl_connect("key_release_event", release)
