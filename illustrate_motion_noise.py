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

from locdemo.draw import HEADING_CMAP, draw_heading_wheel, heading_line, robot_outline
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


def random_walk_states(x0, y0, a0_deg, n, n_steps, step_std_xy, step_halfwidth_deg, rng):
    """A genuine random walk: an independent x/y/heading kick each step,
    accumulated -- unlike scaling one fixed random draw up smoothly over
    time (which makes every particle glide outward in a straight line, like
    a detonation), this actually wanders, the same way propagate_states's
    per-step noise does for the kinematic cases. x, y and heading are drawn
    independently of each other and of position, matching "random, and in
    particular independent of heading."
    """
    x = np.full(n, x0)
    y = np.full(n, y0)
    a_deg = np.full(n, a0_deg)
    states = [(x, y, a_deg)]
    for _ in range(n_steps):
        x = x + step_std_xy * rng.standard_normal(n)
        y = y + step_std_xy * rng.standard_normal(n)
        a_deg = a_deg + rng.uniform(-step_halfwidth_deg, step_halfwidth_deg, n)
        states.append((x, y, a_deg))
    return states


class CloudArtist:
    """One animated panel: particle cloud coloured by heading, plus robot icon."""

    def __init__(self, ax):
        # cmap is set after construction, not passed to scatter() directly:
        # with no data yet, matplotlib silently drops a cmap argument (and
        # warns that it's doing so) -- it doesn't error, so this was easy to
        # not notice, but it meant every particle was coloured by the
        # default viridis instead of the intended wheel colormap.
        self.scat = ax.scatter([], [], s=6, zorder=2)
        self.scat.set_cmap(HEADING_CMAP)
        # Fixed 0-360, not some panel-specific narrower window: a narrower
        # one would stretch the *whole* wheel across whatever slice of
        # headings this panel happens to reach, which pops more on screen
        # but breaks the legend's "0deg is red" for what's actually shown.
        self.scat.set_clim(0, 360)
        (self.body,) = ax.plot([], [], color="k", lw=2, zorder=5)
        (self.head,) = ax.plot([], [], color="k", lw=2, zorder=5)

    def set_cloud(self, x, y, heading_deg):
        self.scat.set_offsets(np.column_stack([x, y]))
        self.scat.set_array(np.mod(heading_deg, 360))

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

    a0_deg = np.rad2deg(a0)

    # Random motion: an actual random walk -- an independent x/y/heading kick
    # at each of the 6 steps, accumulated via random_walk_states, the same
    # way propagate_states does for the kinematic cases. (An earlier version
    # of this panel scaled one fixed random draw up smoothly over time
    # instead; that gets the *final* distribution's variance right but makes
    # every particle glide outward in a dead-straight line, like a
    # detonation, which isn't what a random walk actually looks like.)
    # Heading kicks are uniform, not Gaussian: with only 6 steps a Gaussian
    # wide enough to spread noticeably still tapers off towards the far side
    # of the circle, which understates how little "independent of heading"
    # actually tells you -- a real random guess has no preferred direction,
    # and the colour should end up close to flat everywhere, not just wider.
    # x, y and heading are independent of each other and of position --
    # that decoupling is the whole point: the colour ends up scattered with
    # no spatial pattern, unlike the real motion model where heading and
    # position are tightly correlated (that correlation is exactly what
    # produces the banana).
    #
    # The *position* step size is set so the accumulated spread puts the
    # true final pose about 2 std devs out -- there's *some* support there
    # (a few particles reach it), but most of the probability mass is
    # wasted elsewhere. That's the point of this panel: matching the real
    # motion model's own spread (as it was set before) put the true pose
    # several std devs beyond the cloud's edge, i.e. the isotropic guess
    # assigned it essentially zero probability.
    x_true, y_true, _ = nominal[-1]
    dist_to_true = np.hypot(x_true - x0, y_true - y0)
    spread_total = dist_to_true / 2.0
    step_std_xy = spread_total / np.sqrt(n_steps)
    # Each step's heading kick is Uniform(-90, 90) deg; summing 6 of them
    # (central limit theorem) gives an approximately Normal(0, ~127 deg)
    # walk in heading, wide enough that wrapping it onto the circle comes
    # out close to flat rather than concentrated on one side.
    step_heading_halfwidth_deg = 90.0
    states_random = random_walk_states(x0, y0, a0_deg, n, n_steps, step_std_xy,
                                        step_heading_halfwidth_deg, rng)

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
        # A narrow 2nd column for the heading-wheel legend, so
        # constrained_layout reserves its space instead of it overlapping
        # the data axes.
        fig, (ax, wheel_ax) = plt.subplots(1, 2, figsize=(6.6, 5.2),
                                            gridspec_kw={"width_ratios": [1, 0.3]},
                                            constrained_layout=True)
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if path is not None:
            ax.plot(*path, "--", color="0.6", lw=1, zorder=1)
        cloud = CloudArtist(ax)
        draw_heading_wheel(wheel_ax)
        fig.text(0.01, 0.005, "P. Jensfelt, KTH 2026", ha="left", va="bottom",
                  fontsize=7, color="0.6")
        return fig, ax, cloud

    def make_random_panel(xlim, ylim):
        fig, ax, cloud = make_panel("Random motion (independent of heading)", xlim, ylim,
                                     path=(nominal_x, nominal_y))
        ax.plot(x0, y0, "x", color="k", ms=8, mew=2, zorder=5)

        def update(frame):
            t = frame / args.substeps
            xr, yr, headingr = interp(states_random, t)
            cloud.set_cloud(xr, yr, headingr)
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
