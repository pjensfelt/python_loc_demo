#!/usr/bin/env python3
"""Animates what motion-noise *magnitude* does to the prediction step.

All particles start at the *same* heading -- no prior heading uncertainty --
so any curvature that appears is produced entirely by the motion model's own
noise feeding back on itself across several prediction steps (heading noise
in one step changes the heading used to project position in the next one).

Three separate GIFs, all starting from the same tight initial belief and
(for the two kinematic cases) driving the same nominal arc over 6 prediction
steps:

  1. random motion, independent of heading -- an isotropic blur that just
     grows around the start; heading never changes, so the colour never does
     either
  2. VERY SMALL motion noise -- a thin trace that barely leaves the
     deterministic (noiseless) arc, dashed in grey
  3. LARGER motion noise -- the same arc, but thick enough to fan out into
     the familiar "banana"

All three share one heading colour scale, so they're directly comparable
side by side.

    python illustrate_motion_noise.py
    python illustrate_motion_noise.py --gif motion_noise.gif
        -> writes motion_noise_random.gif, motion_noise_small.gif,
           motion_noise_large.gif
"""

import argparse
from pathlib import Path

import numpy as np

from locdemo.draw import heading_line, robot_outline
from locdemo.models import motion_model, sample_motion_noise


def propagate_states(x, y, a, n_steps, D_step, DA_step, td_std, rda_std, rd_std, rng):
    """Apply the single-step motion model n_steps times, keeping every state.

    Returns a list of (x, y, a) arrays of length n_steps + 1, index 0 being
    the (unmoved) start. This is the prediction step iterated -- the Bayes
    filter's nested convolution: each step's noisy heading becomes the
    *next* step's direction of travel, which is the only way heading noise
    can turn into positional spread with this motion model (a single step
    projects position using the heading it started with, not the noisy one
    it ends at -- see motion_model in locdemo/models.py).
    """
    n = len(np.atleast_1d(x))
    states = [(x, y, a)]
    for _ in range(n_steps):
        D, DA = sample_motion_noise(D_step, DA_step, 1.0, td_std, rda_std, rd_std,
                                     size=n, rng=rng)
        x, y, a = motion_model(x, y, a, D, DA)
        states.append((x, y, a))
    return states


def nominal_states(x0, y0, a0, n_steps, D_step, DA_step):
    """The noiseless backbone arc, one (x, y, a) per step, for reference."""
    states = [(x0, y0, a0)]
    x, y, a = x0, y0, a0
    for _ in range(n_steps):
        x, y, a = motion_model(x, y, a, D_step, DA_step)
        states.append((x, y, a))
    return states


def interp(states, t):
    """Linearly interpolate a list of (x, y, a) arrays at fractional step t."""
    n_steps = len(states) - 1
    t = np.clip(t, 0.0, n_steps)
    k = min(int(t), n_steps - 1)
    frac = t - k
    x = (1 - frac) * states[k][0] + frac * states[k + 1][0]
    y = (1 - frac) * states[k][1] + frac * states[k + 1][1]
    a = (1 - frac) * states[k][2] + frac * states[k + 1][2]
    return x, y, a


class CloudArtist:
    """One animated panel: particle cloud coloured by heading, plus robot icon."""

    def __init__(self, ax, clim):
        self.scat = ax.scatter([], [], s=6, cmap="twilight", zorder=2)
        # vmin/vmax at construction time are silently dropped when there is
        # no data yet (matplotlib warns and ignores them), and set_array
        # later would otherwise autoscale per frame -- which would break the
        # shared colour scale every panel needs to be comparable. Setting it
        # explicitly here sticks across every subsequent set_array.
        self.scat.set_clim(*clim)
        (self.body,) = ax.plot([], [], color="k", lw=2, zorder=5)
        (self.head,) = ax.plot([], [], color="k", lw=2, zorder=5)

    def set_cloud(self, x, y, heading_deg):
        self.scat.set_offsets(np.column_stack([x, y]))
        self.scat.set_array(heading_deg)

    def set_robot(self, pose):
        if pose is None:
            self.body.set_data([], [])
            self.head.set_data([], [])
            return
        self.body.set_data(*robot_outline(*pose, 0.3, 0.2))
        self.head.set_data(*heading_line(*pose, 0.6))


def save_gif(fig, path, total_frames, update, fps):
    """Render every frame and write an undithered, colour-stable GIF.

    Capturing frames ourselves rather than via FuncAnimation + PillowWriter
    matters here: PIL's default GIF save quantizes *each frame
    independently* with Floyd-Steinberg dithering, which both speckles flat
    regions (the white background, the grid) and lets the palette drift from
    frame to frame, so a fixed colour scale flickers instead of staying put.
    Building one shared palette from a few sample frames and quantizing
    every frame against *that*, undithered, keeps colours stable and clean
    across the whole animation.
    """
    from PIL import Image

    w, h = fig.canvas.get_width_height()

    def capture():
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        return Image.frombuffer("RGBA", (w, h), buf, "raw", "RGBA", 0, 1).convert("RGB")

    frames = []
    for f in range(total_frames):
        update(f)
        frames.append(capture())

    sample_idx = sorted(set([0, total_frames // 4, total_frames // 2,
                              3 * total_frames // 4, total_frames - 1]))
    strip = Image.new("RGB", (w * len(sample_idx), h))
    for i, idx in enumerate(sample_idx):
        strip.paste(frames[idx], (i * w, 0))
    palette_img = strip.quantize(colors=256)

    quantized = [f.quantize(palette=palette_img, dither=Image.Dither.NONE) for f in frames]
    quantized[0].save(path, save_all=True, append_images=quantized[1:],
                       duration=int(1000 / fps), loop=0, disposal=2)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=1200, help="number of particles")
    ap.add_argument("--substeps", type=int, default=10,
                     help="interpolated sub-frames per prediction step")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--gif", default=None, metavar="FILE.gif",
                     help="save three animated GIFs (_random/_small/_large suffixes) "
                          "instead of showing live windows")
    args = ap.parse_args()

    if args.gif:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    rng = np.random.default_rng(args.seed)
    n = args.n

    # Prior: tight cloud, *identical* heading for every particle.
    x0, y0, a0 = 0.0, 0.0, np.deg2rad(20.0)
    pos_std = 0.12
    x = x0 + pos_std * rng.standard_normal(n)
    y = y0 + pos_std * rng.standard_normal(n)
    a = np.full(n, a0)

    n_steps = 6
    D_step, DA_step = 3.0 / n_steps, np.deg2rad(35.0) / n_steps
    nominal = nominal_states(x0, y0, a0, n_steps, D_step, DA_step)
    nominal_x = np.array([s[0] for s in nominal])
    nominal_y = np.array([s[1] for s in nominal])

    tiny = dict(td_std=0.01, rda_std=0.02, rd_std=0.01)
    large = dict(td_std=0.18, rda_std=0.3, rd_std=0.18)
    states_tiny = propagate_states(x, y, a, n_steps, D_step, DA_step, rng=rng, **tiny)
    states_large = propagate_states(x, y, a, n_steps, D_step, DA_step, rng=rng, **large)

    # Shared colour scale for all three panels, set from the *large*-noise
    # case's final spread of headings. The random-motion panel holds every
    # particle at the same a0, one solid colour throughout -- itself the
    # point, since that case never touches heading at all.
    a_l_deg = np.rad2deg(states_large[-1][2])
    clim = (a_l_deg.mean() - 3 * a_l_deg.std(), a_l_deg.mean() + 3 * a_l_deg.std())
    a0_deg = np.rad2deg(a0)

    # Random motion: no direction, no kinematics -- just an isotropic blur
    # that grows with sqrt(time), each particle's fixed random *direction*
    # scaled by the growing spread, so the animation is smooth rather than
    # re-randomised every frame. Heading wanders the same way, around a0
    # (random motion has no deterministic turn to drift its mean), with a
    # magnitude matched to the large-noise case's final heading spread --
    # and, crucially, drawn from its *own* independent random numbers (za),
    # not from zx/zy. That decoupling is the whole point: the colour ends up
    # scattered with no spatial pattern, unlike the real motion model where
    # heading and position are tightly correlated (that correlation is
    # exactly what produces the banana).
    #
    # The spread is sized so the true final pose sits about 2 std devs out
    # -- there's *some* support there (a few particles reach it), but most
    # of the probability mass is wasted elsewhere. That's the point of this
    # panel: matching the real motion model's own spread (as it was set
    # before) put the true pose several std devs beyond the cloud's edge,
    # i.e. the isotropic guess assigned it essentially zero probability.
    x_true, y_true, _ = nominal[-1]
    dist_to_true = np.hypot(x_true - x0, y_true - y0)
    spread_total = dist_to_true / 2.0
    # 3x the large-noise case's own heading std, i.e. comparable to the full
    # +-3-sigma colour range that case spans -- at 1x the colour barely moved
    # and the "no spatial pattern" contrast this panel is supposed to show
    # was invisible.
    heading_spread_total = 3 * a_l_deg.std()
    zx, zy, za = (rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n))

    pad = 0.6

    def bounds(*extra_x_y):
        xs = [x] + [e[0] for e in extra_x_y]
        ys = [y] + [e[1] for e in extra_x_y]
        ax_ = np.concatenate(xs)
        ay_ = np.concatenate(ys)
        return (ax_.min() - pad, ax_.max() + pad), (ay_.min() - pad, ay_.max() + pad)

    # The kinematic panels (small/large noise) stay framed tightly around the
    # arc, so the banana is easy to read. The random panel needs its own,
    # much wider framing -- that's the whole point of this panel, seeing how
    # far the isotropic blur has to spread just to reach the true pose.
    xlim_kin, ylim_kin = bounds((nominal_x, nominal_y), states_large[-1])
    xlim_rand, ylim_rand = bounds((nominal_x, nominal_y),
                                   (x0 + spread_total * 3.2 * np.array([-1, 1]),
                                    y0 + spread_total * 3.2 * np.array([-1, 1])))

    total_frames = n_steps * args.substeps + 1

    def make_panel(title, xlim, ylim, path=None):
        fig, ax = plt.subplots(figsize=(5.6, 5.2), constrained_layout=True)
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if path is not None:
            ax.plot(*path, "--", color="0.6", lw=1, zorder=1)
        cloud = CloudArtist(ax, clim)
        fig.colorbar(cloud.scat, ax=ax, shrink=0.85, pad=0.02, label="particle heading [deg]")
        fig.text(0.01, 0.005, "P. Jensfelt, KTH 2026", ha="left", va="bottom",
                  fontsize=7, color="0.6")
        return fig, ax, cloud

    def make_random_panel(xlim, ylim):
        fig, ax, cloud = make_panel("Random motion (independent of heading)", xlim, ylim,
                                     path=(nominal_x, nominal_y))
        ax.plot(x0, y0, "x", color="k", ms=8, mew=2, zorder=5)

        def update(frame):
            t = frame / args.substeps
            growth = np.sqrt(min(t, n_steps) / n_steps)
            spread = spread_total * growth
            heading = a0_deg + heading_spread_total * growth * za
            cloud.set_cloud(x0 + spread * zx, y0 + spread * zy, heading)
            # The true robot still drives the same arc here -- random motion
            # is a (bad) guess about *belief*, not a claim that the robot
            # itself moves randomly. Showing it drive normally while the
            # cloud spreads out symmetrically around a point it isn't even
            # heading towards is the point of this panel.
            cloud.set_robot(interp(nominal, t))
            return (cloud.scat, cloud.body, cloud.head)

        return fig, update

    # Two framings of the same random-motion data: cropped to the same view
    # as the kinematic panels, for a direct side-by-side comparison, and the
    # wide view that shows the cloud actually reaching the true pose.
    fig_r_cropped, update_random_cropped = make_random_panel(xlim_kin, ylim_kin)
    fig_r_wide, update_random_wide = make_random_panel(xlim_rand, ylim_rand)

    fig_t, ax_t, cloud_t = make_panel("Very small motion noise", xlim_kin, ylim_kin,
                                       path=(nominal_x, nominal_y))
    fig_l, ax_l, cloud_l = make_panel("Larger motion noise", xlim_kin, ylim_kin,
                                       path=(nominal_x, nominal_y))

    def update_tiny(frame):
        t = frame / args.substeps
        xt, yt, at = interp(states_tiny, t)
        cloud_t.set_cloud(xt, yt, np.rad2deg(at))
        cloud_t.set_robot(interp(nominal, t))
        return (cloud_t.scat, cloud_t.body, cloud_t.head)

    def update_large(frame):
        t = frame / args.substeps
        xl, yl, al = interp(states_large, t)
        cloud_l.set_cloud(xl, yl, np.rad2deg(al))
        cloud_l.set_robot(interp(nominal, t))
        return (cloud_l.scat, cloud_l.body, cloud_l.head)

    panels = [(fig_r_cropped, update_random_cropped, "random_cropped"),
              (fig_r_wide, update_random_wide, "random_wide"),
              (fig_t, update_tiny, "small"),
              (fig_l, update_large, "large")]

    if args.gif:
        base = Path(args.gif)
        for fig, update, suffix in panels:
            path = base.with_name(f"{base.stem}_{suffix}{base.suffix}")
            save_gif(fig, path, total_frames, update, args.fps)
    else:
        anis = []
        for fig, update, _ in panels:
            ani = FuncAnimation(fig, update, frames=total_frames, interval=1000 / args.fps,
                                 blit=False)
            fig._loc_demo_animation = ani
            anis.append(ani)
        plt.show()


if __name__ == "__main__":
    main()
