"""Shared plumbing for the demo programs: argument parsing and the main loop."""

import argparse
import time
from pathlib import Path

import numpy as np

# Where 'S' screenshots land, so repeated presses don't clutter the working
# directory with loose PNGs next to the source files.
SNAPSHOT_DIR = Path("snapshots")


def save_screenshot(fig, ax, demo):
    """Write two PNGs of the current window into SNAPSHOT_DIR: the full
    window, and just the plot axes (including its ticks/labels, but not the
    side panel).

    Named snap_<second-timestamp>_<demo>_win.png / _plot.png, e.g.
    snapshots/snap_20260922183305_pf_win.png -- timestamped to second
    resolution so repeated presses within a demo run sort together without
    colliding (minute resolution silently overwrote any second `S` press
    within the same minute).
    """
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d%H%M%S")
    win_path = SNAPSHOT_DIR / f"snap_{stamp}_{demo}_win.png"
    plot_path = SNAPSHOT_DIR / f"snap_{stamp}_{demo}_plot.png"
    fig.savefig(win_path, dpi=150)

    fig.canvas.draw()
    bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(plot_path, dpi=150, bbox_inches=bbox)

    print(f"wrote {win_path}\nwrote {plot_path}")
    return win_path, plot_path


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
    ap.add_argument("--x0", type=float, default=0.0, help="true robot's starting x [m]")
    ap.add_argument("--y0", type=float, default=0.0, help="true robot's starting y [m]")
    ap.add_argument("--theta0", type=float, default=0.0,
                    help="true robot's starting heading [deg]")
    ap.add_argument("--set", action="append", default=[], metavar="COL.NAME=VALUE",
                    help="preset a tunable, e.g. --set model.rho=1.0 --set true.td=0.1 "
                         "(angles in degrees, negative means off)")
    ap.add_argument("--landmarks", default=None, metavar="1011",
                    help="which landmarks the filter may use, one digit each")
    if particles:
        ap.add_argument("--particles", type=int, default=10000,
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


def apply_start_pose(world, args):
    """Set the TRUE robot's starting pose from --x0/--y0/--theta0.

    Applied on every reset too (see World.reset), not just at launch, so
    'r' replays the same starting mismatch instead of snapping back to the
    origin. The filter's own belief is untouched -- it always starts at its
    usual [0,0,0] -- which is exactly what makes a non-default start_pose a
    clean way to illustrate localization converging from a known offset,
    without needing to press 'd' live.
    """
    if (args.x0, args.y0, args.theta0) == (0.0, 0.0, 0.0):
        return
    world.start_pose = (args.x0, args.y0, np.deg2rad(args.theta0))
    world.reset()


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
