"""Keyboard control.  Replaces the buttons, sliders and popup of create_ui.m.

The sliders were set-point controls -- you set v and the robot kept going --
so the arrow keys nudge v and w up and down rather than driving the robot only
while held.  Space stops.
"""

import numpy as np
import matplotlib as mpl

from .params import DemoState, N_LADDER, V_STEP, V_MAX, W_STEP, W_MAX, TUNABLES

_COMMON_HELP = """
 driving            filter / display        parameters
 -------            ----------------        ----------
 up/down    v +-    r  reset                tab   select next parameter
 left/right w +-    u  uniform              S-tab select previous
 space      stop    d  disturb true pose    >     increase selected
 0          w = 0   enter force an update   <     decrease selected
 1..4  toggle       g  Gaussian overlay     l     model := true (this row)
       landmark     x  extero. noise on/off L     model := true (all rows)
 h     this help
 q     quit"""

_EKF_ONLY = """
                    i  inject noise (EKF)"""

_PF_ONLY = """
                    c  colour by weight     n/N   fewer/more particles
                    p  resampling on/off"""


def help_text(particles: bool = False) -> str:
    """The key list for the current demo; PF and EKF each have a few keys
    the other doesn't, so they get their own copy of the mode-specific line.
    """
    extra = _PF_ONLY if particles else _EKF_ONLY
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


def make_handler(state: DemoState, fig=None, on_help=None, particles=False):
    """Return a matplotlib key_press_event callback bound to `state`.

    `particles` selects which of the PF-only / EKF-only keys are live, so the
    other filter's controls don't show up as active (or in the help) when
    they wouldn't do anything.
    """

    def clamp(value, limit):
        return float(np.clip(value, -limit, limit))

    def handler(event):
        k = event.key
        if k is None:
            return

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
        elif k == "u":
            state.setUniform = True
        elif k == "d":
            state.addDisturbance = True
        elif k == "i" and not particles:
            state.injectNoise = True
        elif k == "enter":
            state.forceUpdate = True

        # ---- toggles -------------------------------------------------
        elif k == "g":
            state.dispGaussApprox = not state.dispGaussApprox
        elif k == "x":
            state.toggle_extero()
        elif k == "c" and particles:
            state.coloredPts = not state.coloredPts
        elif k == "p" and particles:
            state.resample = not state.resample
        elif k in "1234":
            i = int(k) - 1
            if i < len(state.lmask):
                state.lmask[i] = not state.lmask[i]

        # ---- parameter editing ---------------------------------------
        elif k == "tab":
            state.cursor = (state.cursor + 1) % len(TUNABLES)
        elif k in ("shift+tab", "backtab", "ctrl+tab"):
            state.cursor = (state.cursor - 1) % len(TUNABLES)
        elif k == ">":
            t = state.selected
            state.step(t.column, t.name, +1)
        elif k == "<":
            t = state.selected
            state.step(t.column, t.name, -1)
        elif k == "l":
            state.link_selected()
        elif k == "L":
            state.link_all()

        # ---- particle count ------------------------------------------
        elif k == "N" and particles:
            state.n_idx = min(state.n_idx + 1, len(N_LADDER) - 1)
            state.newN = N_LADDER[state.n_idx]
        elif k == "n" and particles:
            state.n_idx = max(state.n_idx - 1, 0)
            state.newN = N_LADDER[state.n_idx]

        # ---- meta -----------------------------------------------------
        elif k == "h":
            print(help_text(particles))
            if on_help is not None:
                on_help()
        elif k == "q":
            state.running = False
            if fig is not None:
                import matplotlib.pyplot as plt
                plt.close(fig)

    return handler


def connect(fig, state: DemoState, on_help=None, particles=False):
    clear_default_keymap()
    fig.canvas.mpl_connect("key_press_event", make_handler(state, fig, on_help, particles))
