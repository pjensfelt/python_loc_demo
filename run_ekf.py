#!/usr/bin/env python3
"""EKF localization demo.

    python run_ekf.py            open the window
    python run_ekf.py --headless --steps 300 --seed 1 --v 0.5 --w 10
    python run_ekf.py --x0 3 --y0 2 --theta0 45   start the true robot away
                                                   from the filter's [0,0,0]
                                                   belief, to illustrate
                                                   localization without 'd'

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
    app.apply_start_pose(world, args)
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
        panel = draw.Panel(fig, flags=[("95%-Gaussian", lambda s: s.dispGaussApprox),
                                       ("extero", lambda s: not s.extero_off),
                                       ("true robot", lambda s: s.showTrueRobot)],
                           absolute=True)
        keys.connect(fig, state, params, ax=ax, demo="ekf", particles=False, absolute=True)
        if not args.snapshot:
            print(keys.help_text(particles=False, absolute=True))

    def step(_frame=0):
        # ---- simulation ------------------------------------------------
        world.step(state)
        rho, phi = world.measure(state)
        # True distance, not the noisy rho reading -- max_rng is a physical
        # sensing limit, so whether a landmark is even detected at all
        # shouldn't hinge on which way that frame's noise happened to fall.
        in_range = np.hypot(world.xt - params.xL, world.yt - params.yL) <= state.maxRange

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
            ekf.update(rho, phi, state, in_range=in_range)

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
        if state.setHome:
            world.start_pose = world.pose
            print("home pose set -- pass this to start here next time:")
            print(f"--x0 {world.xt:.2f} --y0 {world.yt:.2f} --theta0 {np.rad2deg(world.at):.1f}")
            state.setHome = False
        if state.clearHome:
            world.start_pose = (0.0, 0.0, 0.0)
            print("home pose cleared: (0.00, 0.00, 0.0 deg)")
            state.clearHome = False
        if state.injectGPS:
            xg, yg = world.measure_gps(state)
            ekf.gps_update(xg, yg, state.zGpsStd)
            print(f"GPS fix: ({xg:.3f}, {yg:.3f})")
            state.injectGPS = False
        if state.injectCompass:
            a_meas = world.measure_compass(state)
            ekf.compass_update(a_meas, state.zCompassStd)
            print(f"compass fix: {np.rad2deg(a_meas):.1f} deg")
            state.injectCompass = False

        if fig is None:
            return []

        # ---- drawing ---------------------------------------------------
        true_robot.set_pose(*world.pose)
        true_robot.set_visible(state.showTrueRobot)
        estimate.set(ekf.X[0], ekf.X[1])
        rays.set(world.pose, rho, phi, state.lmask,
                 state.use_range or state.use_bearing, in_range=in_range)
        gauss.set_visible(state.dispGaussApprox)
        if state.dispGaussApprox:
            gauss.set(ekf.X[:2], ekf.P[:2, :2], ekf.X[2], np.sqrt(ekf.P[2, 2]))
        err = np.hypot(ekf.X[0] - world.xt, ekf.X[1] - world.yt)
        panel.update(state, params, f"error   = {err:.3f} m\nsig_x,y = "
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
