#!/usr/bin/env python3
"""Illustrates the prediction step as a convolution, for the lecture slide.

Three panels, all starting from the same prior particle cloud:

  1. the prior belief          p(x_k   | Z_k, U_k)
  2. what "smearing" would look like if the motion model were random noise,
     independent of the robot's heading
  3. what the actual (kinematic) motion model does to it -- the prior gets
     both shifted in the direction of travel and warped into a "banana",
     because particles with different headings project forward differently

    python illustrate_prediction.py
    python illustrate_prediction.py --snapshot prediction_convolution.png
"""

import argparse

import numpy as np

from locdemo.draw import heading_line, robot_outline
from locdemo.models import motion_model, sample_motion_noise, wrap_angle


def sample_prior(n, x0, y0, a0, pos_std, a_std, rng):
    x = x0 + pos_std * rng.standard_normal(n)
    y = y0 + pos_std * rng.standard_normal(n)
    a = wrap_angle(a0 + a_std * rng.standard_normal(n))
    return x, y, a


def draw_panel(ax, x, y, heading_deg, clim, title, robot_pose=None, center_xy=None):
    if heading_deg is None:
        sc = ax.scatter(x, y, s=6, color="0.55", alpha=0.7)
    else:
        sc = ax.scatter(x, y, s=6, c=heading_deg, cmap="twilight", vmin=clim[0], vmax=clim[1])
    if robot_pose is not None:
        ax.plot(*robot_outline(*robot_pose, 0.3, 0.2), color="k", lw=2, zorder=5)
        ax.plot(*heading_line(*robot_pose, 0.6), color="k", lw=2, zorder=5)
    if center_xy is not None:
        ax.plot(*center_xy, "x", color="k", ms=8, mew=2, zorder=5)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    return sc


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=1500, help="number of particles")
    ap.add_argument("--snapshot", default=None, metavar="FILE.png",
                     help="save the figure instead of showing it")
    args = ap.parse_args()

    if args.snapshot:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(args.seed)

    # Prior belief: a little uncertainty in position *and* heading. The
    # heading spread is what turns into curvature once we drive: particles
    # that already think they're pointing slightly differently project
    # forward to different places.
    x0, y0, a0 = 0.0, 0.0, np.deg2rad(20.0)
    pos_std, a_std = 0.12, np.deg2rad(10.0)
    x, y, a = sample_prior(args.n, x0, y0, a0, pos_std, a_std, rng)
    a_deg = np.rad2deg(a)
    clim = (np.rad2deg(a0) - 3 * np.rad2deg(a_std), np.rad2deg(a0) + 3 * np.rad2deg(a_std))

    # Actual motion model: drive forward and turn, same equations and noise
    # sources as locdemo.pf.ParticleFilter.predict.
    D_nom, DA_nom = 3.0, np.deg2rad(35.0)
    td_std, rda_std, rd_std = 0.08, 0.12, 0.08
    D, DA = sample_motion_noise(D_nom, DA_nom, 1.0, td_std, rda_std, rd_std,
                                 size=args.n, rng=rng)
    xk, yk, ak = motion_model(x, y, a, D, DA)
    x_nom, y_nom, a_nom = motion_model(x0, y0, a0, D_nom, DA_nom)

    # If motion were random noise instead: no directional drift, so the
    # cloud stays centred where it was and just blurs out symmetrically.
    # Spread chosen to match the actual motion model's total spread, so the
    # two panels are a fair size-for-size comparison -- the only difference
    # on display is direction, not magnitude.
    spread = np.sqrt(0.5 * (np.var(xk) + np.var(yk)))
    xr = x0 + spread * rng.standard_normal(args.n)
    yr = y0 + spread * rng.standard_normal(args.n)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharex=True, sharey=True,
                              constrained_layout=True)

    draw_panel(axes[0], x, y, a_deg, clim,
               "Prior belief\n$p(x_k\\,|\\,Z_k,U_k)$",
               robot_pose=(x0, y0, a0))
    draw_panel(axes[1], xr, yr, None, clim,
               "If motion were random noise\n(independent of heading)",
               center_xy=(x0, y0))
    sc2 = draw_panel(axes[2], xk, yk, a_deg, clim,
                      "Actual motion model\n$p(x_{k+1}\\,|\\,u_{k+1},x_k)$",
                      robot_pose=(x_nom, y_nom, a_nom))

    pad = 0.6
    all_x = np.concatenate([x, xr, xk])
    all_y = np.concatenate([y, yr, yk])
    axes[0].set_xlim(all_x.min() - pad, all_x.max() + pad)
    axes[0].set_ylim(all_y.min() - pad, all_y.max() + pad)

    fig.colorbar(sc2, ax=axes, shrink=0.75, pad=0.02,
                 label="particle heading [deg]")
    fig.suptitle("Prediction step: convolve the prior with the motion model", fontsize=13)
    fig.text(0.01, 0.01, "P. Jensfelt, KTH 2026", ha="left", va="bottom",
              fontsize=7, color="0.6")

    if args.snapshot:
        fig.savefig(args.snapshot, dpi=200)
        print(f"wrote {args.snapshot}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
