#!/usr/bin/env python3
"""Odometry demos: plain driving, true pose vs. odometry, and a Monte Carlo
simulation of odometry drift, selected with a --mode switch.

    python run_drive.py --mode plain        just drive a robot around
    python run_drive.py --mode uncertain    true pose vs one noisy odometry estimate
    python run_drive.py --mode montecarlo   true pose vs a cloud of noisy estimates

The noise used for the estimates is the MODEL column of the parameter table;
the noise on the true robot's own motion is the TRUE column, which is zero by
default.  Press 'h' for the key bindings.
"""

import numpy as np

from locdemo import app, draw, keys, models
from locdemo.params import Params, DemoState
from locdemo.world import World


def main():
    ap = app.common_args(__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("plain", "uncertain", "montecarlo"),
                    default="uncertain")
    ap.add_argument("--samples", type=int, default=1000,
                    help="number of odometry samples in montecarlo mode")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    params = Params()
    state = DemoState(dispGaussApprox=False)
    # Large model noise so the drift is easy to see from the back of the room.
    for name in ("td", "rda", "rd"):
        state.set_value("model", name, 0.5)
    app.apply_common_args(state, args)

    world = World(params, rng=rng)

    n = args.samples if args.mode == "montecarlo" else 1
    X = np.zeros((3, n))

    fig = None
    if not args.headless:
        plt = app.pyplot(args)
        keys.clear_default_keymap()
        title = {"plain": "Driving a robot",
                 "uncertain": "True pose vs odometry",
                 "montecarlo": "Monte Carlo simulation of odometry"}[args.mode]
        fig = plt.figure(title, figsize=(11, 7))
        ax = draw.setup_axes(fig, params, title)
        true_robot = draw.RobotArtist(ax, params, color="k")
        odo_robot = draw.RobotArtist(ax, params, color="b") if args.mode == "uncertain" else None
        cloud = draw.ParticleArtist(ax, rng=rng) if args.mode == "montecarlo" else None
        panel = draw.Panel(fig)
        keys.connect(fig, state, params)
        if not args.snapshot:
            print(keys.HELP)

    def step(_frame=0):
        world.step(state)

        if args.mode != "plain":
            v_scale, w_scale = models.odometry_scale(
                state.value("true", "r"), state.value("true", "B"), params.r, params.B)
            D, DA = models.sample_motion_noise(
                state.tspeed * v_scale, state.rspeed * w_scale, params.dT,
                state.value("model", "td"),
                state.value("model", "rda"),
                state.value("model", "rd"),
                size=n, rng=rng)
            X[0], X[1], X[2] = models.motion_model(X[0], X[1], X[2], D, DA)

        if state.reset:
            world.reset()
            X[:] = 0.0
            state.reset = False
        if state.addDisturbance:
            world.disturb()
            state.addDisturbance = False

        if fig is None:
            return []

        true_robot.set_pose(*world.pose)
        if odo_robot is not None:
            odo_robot.set_pose(X[0, 0], X[1, 0], X[2, 0])
        if cloud is not None:
            cloud.set(X, np.ones(n), False)
        panel.update(state, params, f"mode = {args.mode}")
        artists = true_robot.artists + panel.artists
        artists += odo_robot.artists if odo_robot else []
        artists += cloud.artists if cloud else []
        return artists

    app.run(fig, state, step, params.dT, args.headless, args.steps, args.snapshot)

    if args.headless:
        print(f"true = {np.round(world.pose, 4)}")
        print(f"odom mean = {np.round(X[:, 0] if n == 1 else X.mean(axis=1), 4)}")


if __name__ == "__main__":
    main()
