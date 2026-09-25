"""Drawing helpers.

Each artist is created once and only its data is updated on every frame,
rather than being deleted and replotted from scratch -- which is what makes
100k particles redraw at 10 Hz.
"""

from itertools import accumulate

import matplotlib
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, SymLogNorm

from .params import (Params, DemoState, TUNABLES, PARAM_ROWS, FIRE_ONCE_ROWS,
                     FIXED_MODEL_ROWS, SINGLE_ROWS, row_is_relevant)

# --------------------------------------------------------------------------
# Heading colour wheel
# --------------------------------------------------------------------------
#
# Same construction as the colour wheel behind flow_to_rgb in
# KTH-RPL/OpenSceneFlow (src/utils/mics.py), so a heading here and a flow
# direction there read the same way: 0 rad (+x) is red, then
# counterclockwise through yellow, green, cyan, blue, magenta and back to
# red. Unlike that function this only ever encodes a direction (no
# magnitude/confidence to fold into brightness), so it's just the wheel
# itself used as a fixed, cyclic colormap over [0, 2*pi).
_WHEEL_TRANSITIONS = (15, 6, 4, 11, 13, 6)


def _make_colorwheel(transitions=_WHEEL_TRANSITIONS):
    n = sum(transitions)
    base_hues = map(np.array, ([255, 0, 0], [255, 255, 0], [0, 255, 0],
                                [0, 255, 255], [0, 0, 255], [255, 0, 255], [255, 0, 0]))
    wheel = np.zeros((n, 3))
    hue_from = next(base_hues)
    start = 0
    for hue_to, end in zip(base_hues, accumulate(transitions)):
        wheel[start:end] = np.linspace(hue_from, hue_to, end - start, endpoint=False)
        hue_from = hue_to
        start = end
    return np.vstack([wheel, [[255, 0, 0]]]) / 255.0  # close the loop back to red


HEADING_CMAP = LinearSegmentedColormap.from_list("direction_wheel", _make_colorwheel(), N=256)


def heading_wheel_image(cmap=None, n=201):
    """RGBA disc: angle -> cmap, masked to a circle, transparent outside it.

    Default cmap is HEADING_CMAP; pass a different one (e.g. matplotlib's
    "twilight") to legend a scatter that was coloured with that instead --
    the wheel is just a picture of whatever cyclic colormap is in use, not
    tied to one specific convention.
    """
    if cmap is None:
        cmap = HEADING_CMAP
    elif isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]
    lin = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(lin, lin)
    theta = np.mod(np.arctan2(yy, xx), 2 * np.pi)
    rgba = np.asarray(cmap(theta / (2 * np.pi)))
    rgba[..., 3] = (np.hypot(xx, yy) <= 1.0).astype(float)
    return rgba


def draw_heading_wheel(ax, cmap=None, n=201):
    """Draw a heading colour wheel as a small legend disc on `ax`.

    A plain linear colorbar makes you read a number off an axis and
    remember what angle that was; a wheel lets you match a particle's
    colour straight to a direction by eye, the same way a compass rose
    does -- which is the point of colouring by heading in the first place.
    Convention: 0 rad (+x, east) at the right, going counterclockwise, same
    as heading_line/robot_outline draw it.
    """
    ax.imshow(heading_wheel_image(cmap, n), extent=(-1, 1, -1, 1), origin="lower", zorder=1)
    for deg, dx, dy, ha, va in [(0, 1.2, 0, "left", "center"),
                                (90, 0, 1.2, "center", "bottom"),
                                (180, -1.2, 0, "right", "center"),
                                (270, 0, -1.2, "center", "top")]:
        ax.text(dx, dy, f"{deg}°", ha=ha, va=va, fontsize=7, color="0.3")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")


def robot_outline(x, y, a, length, width):
    """Closed rectangle outline of the robot at pose (x, y, a)."""
    X = np.array([[-0.5, 0.5, 0.5, -0.5, -0.5],
                  [-0.5, -0.5, 0.5, 0.5, -0.5]]) * np.array([[length], [width]])
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    X = R @ X
    return x + X[0], y + X[1]


def heading_line(x, y, a, length=0.5):
    return [x, x + length * np.cos(a)], [y, y + length * np.sin(a)]


def gauss_ellipse(mu, Sigma, k=np.sqrt(6.0), n=100):
    """Level curve of a 2D Gaussian; k=sqrt(6) matches a ~95% confidence ellipse."""
    vals, vecs = np.linalg.eigh(Sigma)
    vals = np.maximum(vals, 0.0)
    t = np.linspace(0, 2 * np.pi, n)
    w = (k * vecs * np.sqrt(vals)) @ np.vstack([np.cos(t), np.sin(t)])
    return mu[0] + w[0], mu[1] + w[1]


class RobotArtist:
    """Robot outline plus heading vector, drawn as two lines."""

    def __init__(self, ax, params, color="k", lw=2, heading=True):
        self.p = params
        (self.body,) = ax.plot([], [], color=color, lw=lw, zorder=5)
        self.head = None
        if heading:
            (self.head,) = ax.plot([], [], color=color, lw=lw, zorder=5)

    def set_pose(self, x, y, a):
        self.body.set_data(*robot_outline(x, y, a, self.p.length, self.p.width))
        if self.head is not None:
            self.head.set_data(*heading_line(x, y, a))

    def set_visible(self, v):
        for h in self.artists:
            h.set_visible(v)

    @property
    def artists(self):
        return [self.body] + ([self.head] if self.head is not None else [])


class GaussArtist:
    """Covariance ellipse, mean marker and the two heading-uncertainty rays.

    The 'wedge' spanned by the two rays shows +-3 sigma in heading, which is
    the lower right element of P.
    """

    def __init__(self, ax, color="b"):
        (self.ellipse,) = ax.plot([], [], color=color, lw=1, zorder=6)
        (self.mean,) = ax.plot([], [], "x", color=color, ms=6, mew=1.5, zorder=6)
        (self.dir_lo,) = ax.plot([], [], color=color, lw=1, zorder=6)
        (self.dir_hi,) = ax.plot([], [], color=color, lw=1, zorder=6)

    def set(self, mu, Sigma, a, a_std=None, dir_len=0.5):
        self.ellipse.set_data(*gauss_ellipse(np.asarray(mu[:2]), Sigma))
        self.mean.set_data([mu[0]], [mu[1]])
        if a_std is None:
            self.dir_lo.set_data(*heading_line(mu[0], mu[1], a, dir_len))
            self.dir_hi.set_data([], [])
        else:
            self.dir_lo.set_data(*heading_line(mu[0], mu[1], a - 3 * a_std, dir_len))
            self.dir_hi.set_data(*heading_line(mu[0], mu[1], a + 3 * a_std, dir_len))

    def set_visible(self, v):
        for h in self.artists:
            h.set_visible(v)

    @property
    def artists(self):
        return [self.ellipse, self.mean, self.dir_lo, self.dir_hi]


class RayArtist:
    """The magenta lines from the true robot to each measured landmark.

    A landmark beyond `max_rng` is drawn as a short floating dash pointing
    the right direction rather than the full line to it or no line at all --
    either of those would look identical to "this landmark is off" (mask[k]
    False), when out-of-range is a different thing: the sensor is enabled
    for it, it's just too far away to actually return a reading right now.
    Floating a short way out from the robot, rather than starting right at
    it, keeps it from getting lost among the true-robot marker and the
    heading/angle-error lines that also live right at the robot's own
    position.
    """

    STUB_NEAR = 0.8  # m -- where the dash starts
    STUB_FAR = 1.2   # m -- where it ends; short enough to never look real

    def __init__(self, ax, color="m", lw=1):
        self.lc = LineCollection([], colors=color, linewidths=lw, zorder=3)
        ax.add_collection(self.lc)

    def set(self, pose, rho, phi, mask, enabled, in_range=None):
        if not enabled:
            self.lc.set_segments([])
            return
        xt, yt, at = pose
        segs = []
        for k in range(len(rho)):
            if not mask[k]:
                continue
            c, s = np.cos(at + phi[k]), np.sin(at + phi[k])
            if in_range is None or in_range[k]:
                segs.append([(xt, yt), (xt + rho[k] * c, yt + rho[k] * s)])
            else:
                near = min(self.STUB_NEAR, rho[k])
                far = min(self.STUB_FAR, rho[k])
                segs.append([(xt + near * c, yt + near * s), (xt + far * c, yt + far * s)])
        self.lc.set_segments(segs)

    @property
    def artists(self):
        return [self.lc]


class ParticleArtist:
    """Particle cloud, in one of three modes: plain, coloured by weight, or
    coloured by heading.

    Above `max_draw` particles only a random subset is drawn; the filter still
    uses all of them.  The weight colour scale is rescaled every frame --
    essential for the likelihood visualisation, where the absolute weights
    are meaningless but their relative size is not. The heading colour scale
    is the opposite: fixed at [0, 2*pi) always, using HEADING_CMAP, so a given
    colour means the same direction on every frame and every demo run --
    exactly what you want when comparing "did the spread follow the true
    heading" across frames instead of reading relative weight.

    Which *column indices* make up that subset is only re-rolled when the
    particle count changes (a resize), not on every frame. predict/update/
    resample all keep writing to the same N columns -- particle index 3 is
    still particle 3 after a motion step, and still refers to whatever
    survived there after a resample -- so a fixed set of indices already
    tracks it correctly and just shows it move or occasionally get replaced
    by a fitter neighbour. Drawing a *fresh* random subset every frame instead
    would show unrelated particles each time, which looks like flicker even
    though nothing filter-wise is wrong -- rerolling only matters once the
    column space itself has a different size to sample from.
    """

    def __init__(self, ax, max_draw=20000, rng=None):
        self.max_draw = max_draw
        self.rng = np.random.default_rng() if rng is None else rng
        # cmap is set after construction, not passed to scatter() directly --
        # with no data yet, matplotlib warns that it's ignoring cmap (it
        # isn't; set_array() below still finds it) and passing it this way
        # sidesteps the spurious warning entirely.
        self.scat = ax.scatter([], [], s=4, zorder=2)
        self.scat.set_cmap("viridis")
        (self.plain,) = ax.plot([], [], "b.", ms=2, zorder=2)
        self._sel = None
        self._n = None

    def set(self, X, w, mode, draw_all=False):
        """mode is 'plain', 'weight' or 'heading'.

        `draw_all` bypasses the max_draw subsampling entirely -- slow at a
        million particles, but some demos (e.g. watching every particle
        survive or die at once during resampling) are hard to read from a
        20000-particle subset.
        """
        n = X.shape[1]
        if not draw_all and n > self.max_draw:
            if self._sel is None or self._n != n:
                self._sel = self.rng.choice(n, self.max_draw, replace=False)
                self._n = n
            X, w = X[:, self._sel], w[self._sel]
        else:
            self._sel = None
            self._n = n
        self.scat.set_visible(mode != "plain")
        self.plain.set_visible(mode == "plain")
        if mode == "plain":
            self.plain.set_data(X[0], X[1])
            return
        self.scat.set_offsets(np.column_stack([X[0], X[1]]))
        if mode == "heading":
            self.scat.set_cmap(HEADING_CMAP)
            self.scat.set_array(np.mod(X[2], 2 * np.pi))
            self.scat.set_clim(0.0, 2 * np.pi)
        else:  # "weight"
            self.scat.set_cmap("viridis")
            self.scat.set_array(w)
            lo, hi = float(w.min()), float(w.max())
            self.scat.set_clim(lo, hi if hi > lo else lo + 1e-12)

    @property
    def artists(self):
        return [self.scat, self.plain]


class LandmarkArtist:
    """Covariance ellipse and mean marker for one mapped landmark.

    Unlike GaussArtist this has no heading, since a landmark is just a 2D
    point.
    """

    def __init__(self, ax, color="r"):
        (self.ellipse,) = ax.plot([], [], color=color, lw=1, zorder=6)
        (self.mean,) = ax.plot([], [], "o", color=color, ms=5, mfc="none", mew=1.5, zorder=6)

    def set(self, mu, Sigma, show_ellipse):
        self.mean.set_data([mu[0]], [mu[1]])
        self.ellipse.set_visible(show_ellipse)
        if show_ellipse:
            self.ellipse.set_data(*gauss_ellipse(np.asarray(mu), Sigma))

    def set_visible(self, v):
        self.mean.set_visible(v)
        if not v:
            self.ellipse.set_visible(False)

    @property
    def artists(self):
        return [self.ellipse, self.mean]


class LandmarkMapArtist:
    """The growing set of mapped landmarks.

    Landmarks join the state one at a time as EKFSLAM first observes them,
    so artists are created lazily, and only ever as many as have been
    mapped so far are shown.
    """

    def __init__(self, ax, color="r"):
        self.ax = ax
        self.color = color
        self._landmarks = []

    def set(self, mapped, show_ellipse):
        """`mapped` is a list of (mean, covariance) pairs, one per landmark."""
        while len(self._landmarks) < len(mapped):
            self._landmarks.append(LandmarkArtist(self.ax, self.color))
        for i, art in enumerate(self._landmarks):
            if i < len(mapped):
                mu, Sigma = mapped[i]
                art.set_visible(True)
                art.set(mu, Sigma, show_ellipse)
            else:
                art.set_visible(False)

    @property
    def artists(self):
        out = []
        for art in self._landmarks:
            out += art.artists
        return out


class CovarianceArtist:
    """Heatmap of a filter's full covariance matrix, in the same corner
    spot PF's heading-wheel legend uses -- fixed at 3x3 for EKF, growing
    from 3x3 to 3+2*NL for EKF-SLAM as landmarks join the state.

    Colour is sign(P)*log10(sqrt(|P|)), not raw P: position variance (m^2),
    heading variance (rad^2) and landmark blocks live on wildly different
    scales, which would bury the actual structure -- which cells are big,
    which are small, which are positive or negative -- under whichever
    block happens to have the biggest raw numbers on a linear scale. The
    signed log-of-sqrt compresses that range while keeping sign, scaled
    each frame to that frame's own largest magnitude (there's no fixed
    bound the way a correlation matrix's -1..1 would give for free), so a
    diverging colormap centred at 0 still reads directly -- including on
    the diagonal, which *does* vary cell to cell here (unlike a
    correlation matrix, where it's trivially 1.0 everywhere): a bigger
    sig_x than sig_y is visibly a stronger colour, not just a bigger
    printed number. Note this makes the scale genuinely logarithmic: a
    tiny but nonzero cell reads as *strongly* coloured, not faintly, since
    log10 of something near zero is large in magnitude -- only an exact
    0.0 stays neutral. The printed values themselves (see show_values) are
    always the real, untransformed P[i, j].

    Rows/columns are drawn in the state vector's own physical order --
    the robot's 3, then each landmark's 2 in *first-observed* order, not
    landmark-id order, since that's what the real matrix looks like and
    landmarks don't all get mapped in id order. Each landmark block is
    labelled with its id (1-4), matching the numbers already drawn next
    to the true landmark dots on the map, so there's no separate legend
    to cross-reference.
    """

    def __init__(self, ax, cbar_ax=None, show_values=False):
        self.ax = ax
        self.im = None
        self.lines = []
        self.texts = []
        # Printing every cell's value only makes sense at EKF's fixed 3x3 --
        # EKF-SLAM's grid grows well past what 2-decimal numbers can fit
        # without turning into clutter, so this is opt-in per instance,
        # not automatic.
        self.show_values = show_values
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if cbar_ax is not None:
            # A fixed ScalarMappable, decoupled from self.im's actual
            # vmin/vmax (which now rescales every frame to that frame's
            # own largest |sign(P)*log10(sqrt(|P|))| -- see set() below) and from
            # self.im itself being recreated whenever the matrix grows.
            # No ticks and no data-derived bound: ordering (which end is
            # negative, which is positive, and that the middle is zero) is
            # the whole point here, not exact numbers or a scale that would
            # need re-labelling every frame anyway.
            mappable = ScalarMappable(norm=Normalize(vmin=-1.0, vmax=1.0), cmap="coolwarm")
            ax.figure.colorbar(mappable, cax=cbar_ax, ticks=[])
            cbar_ax.set_ylabel("- value +", fontsize=6, labelpad=2)

    def set(self, P, blocks):
        """`blocks` is a list of (label, size) pairs, e.g.
        [("x", 1), ("y", 1), ("θ", 1), ("2", 2), ("1", 2)] -- one
        entry per row/column group, in the matrix's own physical order,
        sizes summing to len(P).
        """
        n = P.shape[0]
        # SymLogNorm on raw P, not a hand-rolled transform: variance/
        # covariance entries can span many orders of magnitude (a 0.05
        # diagonal next to a 1e-5 off-diagonal), which would leave
        # everything but the single largest cell looking flat under a
        # linear colour scale -- but a plain signed log blows up for a
        # genuinely negligible cell (log of near-zero is large in
        # magnitude, so "basically zero" would look strongly coloured).
        # SymLogNorm is linear within +-linthresh of zero (negligible
        # cells stay pale) and logarithmic beyond it (a much bigger cell
        # still reads as bigger, without swamping the rest the way a
        # linear scale would). linthresh and vmax both scale with this
        # frame's own largest magnitude, since there's no fixed bound the
        # way correlation's -1..1 gave for free.
        vmax = max(np.max(np.abs(P)), 1e-9)
        norm = SymLogNorm(linthresh=vmax * 1e-3, vmin=-vmax, vmax=vmax)

        for artist in self.lines + self.texts:
            artist.remove()
        self.lines, self.texts = [], []

        if self.im is None or self.im.get_array().shape != (n, n):
            if self.im is not None:
                self.im.remove()
            # "coolwarm" (Moreland's diverging map), not RdBu_r: RdBu_r is a
            # handful of ColorBrewer stops interpolated between, which can
            # read as distinct colour patches rather than one flowing
            # gradient -- coolwarm is built specifically for a smooth,
            # perceptually continuous path from blue through a neutral
            # midpoint to red.
            self.im = self.ax.imshow(P, cmap="coolwarm", norm=norm,
                                     origin="upper", extent=(0, n, n, 0), zorder=1)
        else:
            self.im.set_data(P)
            self.im.set_norm(norm)

        if self.show_values:
            # Every printed number is P[i, j] exactly as it is -- variance
            # on the diagonal, raw covariance off it -- not whatever
            # SymLogNorm maps it to for the *colour*.
            for i in range(n):
                for j in range(n):
                    # White on the strongly-coloured cells, black on the
                    # pale ones near this frame's zero -- coolwarm's centre
                    # is light enough that a single fixed colour would be
                    # illegible on one half or the other. norm(x) lands in
                    # [0, 1] with 0.5 at zero, so distance from 0.5 is how
                    # far this cell sits from the middle of the scale.
                    color = "w" if abs(norm(P[i, j]) - 0.5) > 0.3 else "k"
                    self.texts.append(self.ax.text(
                        j + 0.5, i + 0.5, f"{P[i, j]:.2g}",
                        ha="center", va="center", fontsize=6, color=color, zorder=3))

        pos = 0
        for label, size in blocks:
            if pos > 0:
                self.lines.append(self.ax.axhline(pos, color="k", lw=0.6, zorder=2))
                self.lines.append(self.ax.axvline(pos, color="k", lw=0.6, zorder=2))
            mid = pos + size / 2
            self.texts.append(self.ax.text(-0.4, mid, label, ha="right", va="center",
                                            fontsize=7))
            self.texts.append(self.ax.text(mid, -0.4, label, ha="center", va="bottom",
                                            fontsize=7))
            pos += size
        self.ax.set_xlim(0, n)
        self.ax.set_ylim(n, 0)

    @property
    def artists(self):
        out = [self.im] if self.im is not None else []
        return out + self.lines + self.texts


class PointArtist:
    """A single estimated position, drawn as a dot."""

    def __init__(self, ax, color="b", ms=6):
        (self.pt,) = ax.plot([], [], ".", color=color, ms=ms, zorder=6)

    def set(self, x, y):
        self.pt.set_data([x], [y])

    @property
    def artists(self):
        return [self.pt]


def setup_axes(fig, params: Params, title):
    """One axes with the landmarks drawn, plus room for the parameter panel."""
    ax = fig.add_axes([0.30, 0.08, 0.68, 0.86])
    ax.set_xlim(*params.xlim)
    ax.set_ylim(*params.ylim)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.plot(params.xL, params.yL, "ko", ms=5, zorder=4)
    offs = [(-0.3, -0.3), (0.3, -0.3), (0.3, 0.3), (-0.3, 0.3)]
    for k in range(params.NL):
        dx, dy = offs[k % len(offs)]
        ax.text(params.xL[k] + dx, params.yL[k] + dy, str(k + 1), ha="center", va="center")
    fig.text(0.99, 0.01, "P. Jensfelt, KTH 2026", ha="right", va="bottom",
              fontsize=7, color="0.6")
    return ax


class Panel:
    """Left-hand text panel: the true/model parameter table and the status."""

    def __init__(self, fig, flags=(), absolute=False, slam=False, pgo=False):
        self.flags = flags
        # Same absolute/slam/pgo flags keys.py wires its keys with -- a row
        # whose key isn't wired for this demo (e.g. sig_cmp where 'y' isn't
        # bound) is blanked out below rather than shown doing nothing.
        self.absolute, self.slam, self.pgo = absolute, slam, pgo
        self.text = fig.text(0.015, 0.97, "", family="monospace", fontsize=9,
                             va="top", ha="left")

    def _relevant(self, name):
        return row_is_relevant(name, absolute=self.absolute, slam=self.slam, pgo=self.pgo)

    def update(self, state: DemoState, params: Params, extra=""):
        rows = ["        TRUE     MODEL", "        ----     -----"]
        for name in PARAM_ROWS:
            cells = []
            for column in ("true", "model"):
                t = next(t for t in TUNABLES if t.key == (column, name))
                sel = TUNABLES.index(t) == state.cursor
                txt = t.format(state.value(column, name))
                cells.append(("[%s]" if sel else " %s ") % txt.center(7))
            label = next(t for t in TUNABLES if t.name == name).label
            rows.append(f"{label:>7} {cells[0]}{cells[1]}")

        # GPS/compass: fire-once sensors (keys 'G'/'y'), not continuous
        # measurements -- whether one happens at all is already controlled
        # by pressing the key, so unlike the rows above there is no "off"
        # state to select here.
        rows.append("        (fire-once sensors)")
        for name in FIRE_ONCE_ROWS:
            if not self._relevant(name):
                rows.append("")
                continue
            cells = []
            for column in ("true", "model"):
                t = next(t for t in TUNABLES if t.key == (column, name))
                sel = TUNABLES.index(t) == state.cursor
                txt = t.format(state.value(column, name))
                cells.append(("[%s]" if sel else " %s ") % txt.center(7))
            label = next(t for t in TUNABLES if t.name == name).label
            rows.append(f"{label:>7} {cells[0]}{cells[1]}")

        # Wheel r/B and compass bias: model is fixed on Params (never
        # selectable, no brackets), only the true hardware value is an
        # editable ladder entry.
        rows.append("        (fixed bias)")
        for name in FIXED_MODEL_ROWS:
            if not self._relevant(name):
                rows.append("")
                continue
            t = next(t for t in TUNABLES if t.key == ("true", name))
            sel = TUNABLES.index(t) == state.cursor
            true_txt = t.format(state.value("true", name))
            true_cell = ("[%s]" if sel else " %s ") % true_txt.center(7)
            model_cell = " %s " % t.format(getattr(params, name)).center(7)
            rows.append(f"{t.label:>7} {true_cell}{model_cell}")

        # Rows with just the one column -- no "model" cell at all, unlike
        # every row above (even FIXED_MODEL_ROWS still shows a second,
        # read-only cell).
        for name in SINGLE_ROWS:
            if not self._relevant(name):
                rows.append("")
                continue
            t = next(t for t in TUNABLES if t.key == ("true", name))
            sel = TUNABLES.index(t) == state.cursor
            txt = t.format(state.value("true", name))
            cell = ("[%s]" if sel else " %s ") % txt.center(7)
            rows.append(f"{t.label:>7} {cell}")

        lm = " ".join(f"L{k+1}" if state.lmask[k] else " . " for k in range(len(state.lmask)))
        rows += [
            "",
            f"v = {state.tspeed:+.2f} m/s",
            f"w = {np.rad2deg(state.rspeed):+.1f} deg/s",
            f"landmarks: {lm}",
        ]
        for label, value in self.flags:
            v = value(state)
            rows.append(f"{label}: {('on' if v else 'off') if isinstance(v, bool) else v}")
        if extra:
            rows += ["", extra]
        rows += ["", "press 'h' for keys"]
        self.text.set_text("\n".join(rows))

    @property
    def artists(self):
        return [self.text]
