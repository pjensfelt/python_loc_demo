#!/usr/bin/env python3
"""Monte Carlo Localization demo (particle filter).

    python run_pf.py --particles 1000
    python run_pf.py --headless --steps 300 --seed 1 --v 0.5 --set model.rho=1.0

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
    # Deliberately larger motion noise than the EKF's default, so the
    # particles spread out and explore the state space.
    for name in ("td", "rda", "rd"):
        state.set_value("model", name, 0.25)
    state.n_idx = min(range(len(N_LADDER)),
                      key=lambda i: abs(N_LADDER[i] - args.particles))
    app.apply_common_args(state, args)

    world = World(params, rng=rng)
    pf = ParticleFilter(params, N=args.particles, rng=rng)

    fig = None
    if not args.headless:
        plt = app.pyplot(args)
        keys.clear_default_keymap()
        fig = plt.figure("Monte Carlo Localization", figsize=(11, 7))
        ax = draw.setup_axes(fig, params, "Monte Carlo Localization")
        particles = draw.ParticleArtist(ax, rng=rng)
        true_robot = draw.RobotArtist(ax, params, color="k")
        gauss = draw.GaussArtist(ax, color="r")
        rays = draw.RayArtist(ax)
        panel = draw.Panel(fig, flags=[("Gaussian", lambda s: s.dispGaussApprox),
                                       ("colour", lambda s: s.coloredPts),
                                       ("resample", lambda s: s.resample),
                                       ("extero", lambda s: not s.extero_off)])
        keys.connect(fig, state, particles=True)
        if not args.snapshot:
            print(keys.help_text(particles=True))

    warned_tiny = [False]
    cached_approx = [None]

    def step(_frame=0):
        # ---- simulation ------------------------------------------------
        world.step(state)
        rho, phi = world.measure(state)

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
            pf.update(rho, phi, state)
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
        if state.newN != pf.N:
            print(f"Resampling particle set, N={state.newN}")
            pf.set_size(state.newN)
            changed = True

        if fig is None:
            return []

        # ---- drawing ---------------------------------------------------
        particles.set(pf.X, pf.w, state.coloredPts)
        true_robot.set_pose(*world.pose)
        rays.set(world.pose, rho, phi, state.lmask,
                 state.use_range or state.use_bearing)

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
        panel.update(state, f"N       = {pf.N}\nsum w   = {pf.w.sum():.3e}\n"
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
