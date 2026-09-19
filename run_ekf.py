#!/usr/bin/env python3
"""EKF localization demo.

    python run_ekf.py            open the window
    python run_ekf.py --headless --steps 300 --seed 1 --v 0.5 --w 10

Press 'h' in the window (or see README.md) for the key bindings.
"""

import numpy as np

from locdemo import app, draw, keys
from locdemo.ekf import EKFLocalizer
from locdemo.params import Params, DemoState
from locdemo.world import World


def main():
    args = app.common_args(__doc__.splitlines()[0]).parse_args()
    rng = np.random.default_rng(args.seed)

    params = Params()
    state = DemoState(dispGaussApprox=True)
    app.apply_common_args(state, args)

    world = World(params, rng=rng)
    ekf = EKFLocalizer(params)

    fig = None
    if not args.headless:
        plt = app.pyplot(args)
        keys.clear_default_keymap()
        fig = plt.figure("EKF localization", figsize=(11, 7))
        ax = draw.setup_axes(fig, params, "EKF localization")
        true_robot = draw.RobotArtist(ax, params, color="k")
        estimate = draw.PointArtist(ax, color="b")
        gauss = draw.GaussArtist(ax, color="b")
        rays = draw.RayArtist(ax)
        panel = draw.Panel(fig, flags=[("Gaussian", lambda s: s.dispGaussApprox),
                                       ("extero", lambda s: not s.extero_off)])
        keys.connect(fig, state, particles=False)
        if not args.snapshot:
            print(keys.help_text(particles=False))

    def step(_frame=0):
        # ---- simulation ------------------------------------------------
        world.step(state)
        rho, phi = world.measure(state)

        # ---- filter ----------------------------------------------------
        if state.injectNoise:
            ekf.inject_noise()
            state.injectNoise = False

        # The filter only runs while the robot is moving, or when an update
        # is forced.  Standing still and updating forever would shrink the
        # covariance far below what the correlated measurements justify.
        if state.forceUpdate or state.moving:
            state.forceUpdate = False
            ekf.predict(state)
            ekf.update(rho, phi, state)

        # ---- one-shot requests ----------------------------------------
        if state.reset:
            world.reset()
            ekf.reset()
            state.reset = False
        if state.setUniform:
            ekf.set_uniform()
            state.setUniform = False
        if state.addDisturbance:
            world.disturb()
            state.addDisturbance = False

        if fig is None:
            return []

        # ---- drawing ---------------------------------------------------
        true_robot.set_pose(*world.pose)
        estimate.set(ekf.X[0], ekf.X[1])
        rays.set(world.pose, rho, phi, state.lmask,
                 state.use_range or state.use_bearing)
        gauss.set_visible(state.dispGaussApprox)
        if state.dispGaussApprox:
            gauss.set(ekf.X[:2], ekf.P[:2, :2], ekf.X[2], np.sqrt(ekf.P[2, 2]))
        err = np.hypot(ekf.X[0] - world.xt, ekf.X[1] - world.yt)
        panel.update(state, f"error   = {err:.3f} m\nsig_x,y = "
                            f"{np.sqrt(ekf.P[0,0]):.3f}, {np.sqrt(ekf.P[1,1]):.3f} m")
        return (true_robot.artists + estimate.artists + gauss.artists
                + rays.artists + panel.artists)

    app.run(fig, state, step, params.dT, args.headless, args.steps, args.snapshot)

    if args.headless:
        print(f"true  = {np.round(world.pose, 4)}")
        print(f"X     = {np.round(ekf.X, 4)}")
        print(f"diag P= {np.round(np.diag(ekf.P), 6)}")


if __name__ == "__main__":
    main()
