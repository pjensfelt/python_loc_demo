"""Shared plumbing for the demo programs: argument parsing and the main loop."""

import argparse
import numpy as np


def common_args(description, particles=False):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--seed", type=int, default=None,
                    help="seed the random generator for a repeatable run")
    ap.add_argument("--headless", action="store_true",
                    help="run without a window, for testing")
    ap.add_argument("--steps", type=int, default=200,
                    help="number of steps to run when headless or snapshotting")
    ap.add_argument("--snapshot", default=None, metavar="FILE.png",
                    help="run --steps steps offscreen and save the final frame")
    ap.add_argument("--v", type=float, default=0.0, help="initial speed [m/s]")
    ap.add_argument("--w", type=float, default=0.0, help="initial turn rate [deg/s]")
    ap.add_argument("--set", action="append", default=[], metavar="COL.NAME=VALUE",
                    help="preset a tunable, e.g. --set model.rho=1.0 --set true.td=0.1 "
                         "(angles in degrees, negative means off)")
    ap.add_argument("--landmarks", default=None, metavar="1011",
                    help="which landmarks the filter may use, one digit each")
    if particles:
        ap.add_argument("--particles", type=int, default=100,
                        help="initial number of particles")
        ap.add_argument("--resample", action="store_true",
                        help="start with resampling enabled")
    return ap


def apply_common_args(state, args):
    """Apply --v/--w/--set/--landmarks so an experiment can be scripted."""
    state.tspeed = args.v
    state.rspeed = np.deg2rad(args.w)

    for item in getattr(args, "set", []):
        spec, _, raw = item.partition("=")
        column, _, name = spec.partition(".")
        value = float(raw)
        if name == "phi" and value > 0:
            value = np.deg2rad(value)
        state.set_value(column, name, value)

    if getattr(args, "resample", False):
        state.resample = True

    if args.landmarks:
        for i, ch in enumerate(args.landmarks[:len(state.lmask)]):
            state.lmask[i] = ch not in "0nN"


def pyplot(args):
    """Import pyplot, forcing a non-interactive backend when snapshotting."""
    import matplotlib
    if args.snapshot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def run(fig, state, step, dT, headless=False, steps=0, snapshot=None):
    """Drive `step` either from a matplotlib timer or as a plain loop."""
    if snapshot:
        for i in range(steps):
            step(i)
        fig.savefig(snapshot, dpi=110)
        print(f"wrote {snapshot}")
        return

    if headless:
        for i in range(steps):
            step(i)
            if not state.running:
                break
        return

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # Held in a local so the animation is not garbage collected.
    ani = FuncAnimation(fig, step, interval=int(dT * 1000),
                        blit=False, cache_frame_data=False)
    fig._loc_demo_animation = ani
    plt.show()
