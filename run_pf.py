#!/usr/bin/env python3
"""Monte Carlo Localization demo (particle filter).

    python run_pf.py --particles 1000
    python run_pf.py --headless --steps 300 --seed 1 --v 0.5 --set model.rho=1.0
    python run_pf.py --x0 3 --y0 2 --theta0 45   start the true robot away
                                                  from the particles' [0,0,0]
                                                  prior, to illustrate
                                                  localization without 'd'

Press 'h' in the window (or see README.md) for the key bindings.
"""

import numpy as np

from locdemo import app, draw, keys
from locdemo.params import Params, DemoState, N_LADDER
from locdemo.pf import ParticleFilter
from locdemo.world import World


def main():
    args = app.common_args(__doc__.splitlines()[0], particles=True).parse_args()
    rng = np.random.default_rng(args.seed)

    params = Params()
    state = DemoState(newN=args.particles)
    state.n_idx = min(range(len(N_LADDER)),
                      key=lambda i: abs(N_LADDER[i] - args.particles))
    app.apply_common_args(state, args)

    world = World(params, rng=rng)
    app.apply_start_pose(world, args)
    pf = ParticleFilter(params, N=args.particles, rng=rng)

    fig = None
    if not args.headless:
        plt = app.pyplot(args)
        keys.clear_default_keymap()
        fig = plt.figure("Monte Carlo Localization", figsize=(11, 7))
        ax = draw.setup_axes(fig, params, "Monte Carlo Localization")
        # Legend for heading colour, tucked under the parameter panel; only
        # relevant (and only shown) while that colour mode is active.
        wheel_ax = fig.add_axes([0.03, 0.02, 0.20, 0.20])
        draw.draw_heading_wheel(wheel_ax)
        particles = draw.ParticleArtist(ax, rng=rng)
        true_robot = draw.RobotArtist(ax, params, color="k")
        gauss = draw.GaussArtist(ax, color="r")
        rays = draw.RayArtist(ax)
        panel = draw.Panel(fig, flags=[("95%-Gaussian", lambda s: s.dispGaussApprox),
                                       ("colour", lambda s: s.ptColorMode),
                                       ("draw all particles", lambda s: s.drawAllParticles),
                                       ("resample", lambda s: s.resample),
                                       ("extero", lambda s: not s.extero_off),
                                       ("true robot", lambda s: s.showTrueRobot)],
                           absolute=True)
        keys.connect(fig, state, params, ax=ax, demo="pf", particles=True, absolute=True)
        if not args.snapshot:
            print(keys.help_text(particles=True, absolute=True))

    warned_tiny = [False]
    cached_approx = [None]

    def step(_frame=0):
        # ---- simulation ------------------------------------------------
        world.step(state)
        rho, phi = world.measure(state)
        # True distance, not the noisy rho reading -- see run_ekf.py.
        in_range = np.hypot(world.xt - params.xL, world.yt - params.yL) <= state.maxRange

        # ---- filter ----------------------------------------------------
        # Tracks whether the particle set changed this frame, so the cached
        # Gaussian overlay (drawn from a fresh random resample every time
        # it's recomputed) is only redone when there's something new to fit.
        changed = False
        if state.forceUpdate or state.moving:
            state.forceUpdate = False

            if pf.maybe_resample(state):
                changed = True

            # With resampling switched off the weights decay without bound and
            # eventually underflow.  In practice resampling prevents this; the
            # warning is here because the demo lets you turn it off.
            if pf.weights_are_tiny():
                if not warned_tiny[0]:
                    print("You should really resample now, or increase the sensor "
                          f"noise (min={pf.w.min():.3e}, max={pf.w.max():.3e})")
                warned_tiny[0] = True
            else:
                warned_tiny[0] = False

            pf.predict(state)
            pf.update(rho, phi, state, in_range=in_range)
            changed = True

        # ---- one-shot requests ----------------------------------------
        if state.reset:
            world.reset()
            pf.reset()
            state.reset = False
            changed = True
        if state.setUniform:
            pf.set_uniform()
            state.setUniform = False
            changed = True
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
        if state.resampleOnce:
            pf.resample_now()
            state.resampleOnce = False
            changed = True
        if state.newN != pf.N:
            print(f"Resampling particle set, N={state.newN}")
            pf.set_size(state.newN)
            changed = True
        if state.injectGPS:
            xg, yg = world.measure_gps(state)
            pf.gps_update(xg, yg, state.zGpsStd)
            print(f"GPS fix: ({xg:.3f}, {yg:.3f})")
            # Reweight only -- no resample here. Resampling happens the same
            # way it does for every other measurement: via maybe_resample()
            # above, next time you press enter or drive (see the top of this
            # function), not as a side effect of firing the GPS itself.
            changed = True
            state.injectGPS = False
        if state.injectCompass:
            a_meas = world.measure_compass(state)
            pf.compass_update(a_meas, state.zCompassStd)
            print(f"compass fix: {np.rad2deg(a_meas):.1f} deg")
            changed = True
            state.injectCompass = False

        if fig is None:
            return []

        # ---- drawing ---------------------------------------------------
        particles.set(pf.X, pf.w, state.ptColorMode, draw_all=state.drawAllParticles)
        wheel_ax.set_visible(state.ptColorMode == "heading")
        true_robot.set_pose(*world.pose)
        true_robot.set_visible(state.showTrueRobot)
        rays.set(world.pose, rho, phi, state.lmask,
                 state.use_range or state.use_bearing, in_range=in_range)

        # gaussian_approx() draws a fresh random sample of the particles, so
        # recomputing it every frame made the overlay jitter even when the
        # particle set itself was static; only redo it when there's something
        # new to fit.
        if state.dispGaussApprox and (changed or cached_approx[0] is None):
            cached_approx[0] = pf.gaussian_approx()
        approx = cached_approx[0] if state.dispGaussApprox else None
        gauss.set_visible(approx is not None)
        if approx is not None:
            mu, sigma, muA = approx
            gauss.set(mu, sigma, muA)

        neff = pf.w.sum() ** 2 / np.sum(pf.w ** 2)
        panel.update(state, params, f"N       = {pf.N}\nsum w   = {pf.w.sum():.3e}\n"
                            f"N_eff   = {neff:.1f}")
        return (particles.artists + true_robot.artists + gauss.artists
                + rays.artists + panel.artists)

    app.run(fig, state, step, params.dT, args.headless, args.steps, args.snapshot)

    if args.headless:
        mu = (pf.X[:2] * pf.w).sum(axis=1) / pf.w.sum()
        print(f"true      = {np.round(world.pose, 4)}")
        print(f"mean xy   = {np.round(mu, 4)}")
        print(f"N, sum w  = {pf.N}, {pf.w.sum():.3e}")


if __name__ == "__main__":
    main()
