#!/usr/bin/env python3
"""EKF-SLAM demo: landmark positions are unknown and get mapped on the fly.

    python run_ekfslam.py
    python run_ekfslam.py --headless --steps 300 --seed 1 --v 0.5 --w 10

Press 'h' in the window (or see README.md) for the key bindings, including
's' for a superGPS fix that is specific to this demo.
"""

import numpy as np

from locdemo import app, draw, keys
from locdemo.ekfslam import EKFSLAM
from locdemo.params import Params, DemoState
from locdemo.world import World


def main():
    args = app.common_args(__doc__.splitlines()[0]).parse_args()
    rng = np.random.default_rng(args.seed)

    params = Params()
    state = DemoState(dispGaussApprox=True)
    # Every other demo defaults model.rho/phi to "off" (dead reckoning
    # first, measurements on deliberately) -- but EKF-SLAM's own update()
    # returns immediately when both are off (see its docstring), so with
    # lmask's own default of all-on this demo did *nothing* with landmarks
    # out of the box: mapped none, fused none. Calibrate the model to the
    # true noise instead (a properly-tuned filter, the "matched" case the
    # EKF demo's own docs point you at with `l`), and default landmarks off
    # instead, matching PGO's own "discover as you go" convention -- so
    # turning one on with '1'..'4' is a deliberate, visible step rather
    # than four landmarks appearing mapped on the very first frame.
    state.set_value("model", "rho", state.value("true", "rho"))
    state.set_value("model", "phi", state.value("true", "phi"))
    # Lower than the shared 0.25 default: EKF-SLAM's own linearisation gets
    # noticeably less consistent at higher motion noise over any real
    # distance (a landmark can end up confidently mapped several metres
    # from where it actually is, well outside its own reported ellipse --
    # not a bug, just how badly a wrong operating point hurts an EKF's own
    # Jacobians), which makes loop closure a much less convincing demo. A
    # smaller, matched value keeps drift real and worth correcting without
    # tipping into that regime.
    for name in ("td", "rda", "rd"):
        state.set_value("true", name, 0.05)
        state.set_value("model", name, 0.05)
    state.lmask[:] = False
    app.apply_common_args(state, args)

    world = World(params, rng=rng)
    app.apply_start_pose(world, args)
    slam = EKFSLAM(params)

    fig = None
    if not args.headless:
        plt = app.pyplot(args)
        keys.clear_default_keymap()
        fig = plt.figure("EKF-SLAM", figsize=(11, 7))
        ax = draw.setup_axes(fig, params, "EKF-SLAM")
        true_robot = draw.RobotArtist(ax, params, color="k")
        estimate = draw.PointArtist(ax, color="b")
        gauss = draw.GaussArtist(ax, color="b")
        landmark_map = draw.LandmarkMapArtist(ax, color="r")
        rays = draw.RayArtist(ax)
        cov_ax = fig.add_axes([0.03, 0.02, 0.20, 0.20])
        cov = draw.CovarianceArtist(cov_ax)
        panel = draw.Panel(fig, flags=[("95%-Gaussian", lambda s: s.dispGaussApprox),
                                       ("ellipses", lambda s: "robot-relative" if s.relativeUncertainty else "world"),
                                       ("extero", lambda s: not s.extero_off),
                                       ("true robot", lambda s: s.showTrueRobot)],
                           slam=True)
        keys.connect(fig, state, params, ax=ax, demo="ekfslam", particles=False, slam=True)
        if not args.snapshot:
            print(keys.help_text(particles=False, slam=True))

    def step(_frame=0):
        # ---- simulation ------------------------------------------------
        world.step(state)
        rho, phi = world.measure(state)
        # True distance, not the noisy rho reading -- see run_ekf.py.
        in_range = np.hypot(world.xt - params.xL, world.yt - params.yL) <= state.maxRange

        # ---- filter ----------------------------------------------------
        if state.injectNoise:
            slam.inject_noise()
            state.injectNoise = False

        # As in the EKF demo, prediction and the landmark update only run
        # while driving or when forced. A superGPS fix on its own still
        # forces a (near no-op) predict step, matching the original demo.
        do_update = state.forceUpdate or state.moving
        if state.forceUpdate or state.superGPS or state.moving:
            state.forceUpdate = False
            slam.predict(state)
            if do_update:
                slam.update(rho, phi, state, in_range=in_range)
            if state.superGPS:
                slam.super_gps_update(world.xt, world.yt)
        state.superGPS = False

        # ---- one-shot requests ----------------------------------------
        if state.reset:
            world.reset()
            slam.reset()
            state.reset = False
        if state.setUniform:
            slam.set_uniform()
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

        if fig is None:
            return []

        # ---- drawing ---------------------------------------------------
        true_robot.set_pose(*world.pose)
        true_robot.set_visible(state.showTrueRobot)
        estimate.set(slam.X[0], slam.X[1])
        # The rays are always-on live sensor feedback -- they reflect what
        # the sensor currently reads regardless of whether the filter uses
        # it this frame. Only predict/update (above) are gated on moving or
        # a forced update.
        rays.set(world.pose, rho, phi, state.lmask,
                 state.use_range or state.use_bearing, in_range=in_range)
        gauss.set_visible(state.dispGaussApprox)
        if state.dispGaussApprox:
            if state.relativeUncertainty:
                # "Relative to the robot" applies to the robot itself too --
                # its own position and heading relative to itself are known
                # exactly (variance zero), by definition, however uncertain
                # they are in the world frame. 1e-6 (not literal 0) matches
                # EKFSLAM.P0's own "known exactly" convention, so the ellipse
                # draws as a tiny but visible dot instead of vanishing.
                gauss.set(slam.X[:2], 1e-6 * np.eye(2), slam.X[2], 1e-6)
            else:
                gauss.set(slam.X[:2], slam.P[:2, :2], slam.X[2], np.sqrt(slam.P[2, 2]))

        mapped = (slam.mapped_landmarks_relative() if state.relativeUncertainty
                 else slam.mapped_landmarks())
        landmark_map.set([(mu, Sigma) for _, mu, Sigma in mapped], state.dispGaussApprox)

        # Blocks in the state vector's own physical order (first-observed,
        # not landmark id) -- labelled by id (1-4) so each block matches
        # the same number already drawn next to that landmark's true dot.
        blocks = [("x", 1), ("y", 1), ("θ", 1)]
        for l, _ in sorted(slam.landmark_index.items(), key=lambda kv: kv[1]):
            blocks.append((str(l + 1), 2))
        cov.set(slam.P, blocks)

        err = np.hypot(slam.X[0] - world.xt, slam.X[1] - world.yt)
        panel.update(state, params, f"error   = {err:.3f} m\nsig_x,y = "
                            f"{np.sqrt(slam.P[0,0]):.3f}, {np.sqrt(slam.P[1,1]):.3f} m\n"
                            f"mapped  = {len(mapped)}/{params.NL}\n"
                            f"state   = {len(slam.X)} (3 + 2 per mapped landmark)")
        return (true_robot.artists + estimate.artists + gauss.artists
                + rays.artists + landmark_map.artists + cov.artists + panel.artists)

    app.run(fig, state, step, params.dT, args.headless, args.steps, args.snapshot)

    if args.headless:
        print(f"true  = {np.round(world.pose, 4)}")
        print(f"X     = {np.round(slam.X, 4)}")
        print(f"diag P= {np.round(np.diag(slam.P), 6)}")
        for l, mu, Sigma in slam.mapped_landmarks():
            true_xy = params.xL[l], params.yL[l]
            print(f"landmark {l+1}: true={np.round(true_xy, 4)} "
                  f"est={np.round(mu, 4)} sig={np.round(np.sqrt(np.diag(Sigma)), 4)}")


if __name__ == "__main__":
    main()
