"""Drawing helpers.

Each artist is created once and only its data is updated on every frame,
rather than being deleted and replotted from scratch -- which is what makes
100k particles redraw at 10 Hz.
"""

import numpy as np
from matplotlib.collections import LineCollection

from .params import Params, DemoState, TUNABLES, PARAM_ROWS, FIXED_MODEL_ROWS


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
    """The magenta lines from the true robot to each measured landmark."""

    def __init__(self, ax, color="m", lw=1):
        self.lc = LineCollection([], colors=color, linewidths=lw, zorder=3)
        ax.add_collection(self.lc)

    def set(self, pose, rho, phi, mask, enabled):
        if not enabled:
            self.lc.set_segments([])
            return
        xt, yt, at = pose
        segs = [[(xt, yt),
                 (xt + rho[k] * np.cos(at + phi[k]), yt + rho[k] * np.sin(at + phi[k]))]
                for k in range(len(rho)) if mask[k]]
        self.lc.set_segments(segs)

    @property
    def artists(self):
        return [self.lc]


class ParticleArtist:
    """Particle cloud, optionally coloured by weight.

    Above `max_draw` particles only a random subset is drawn; the filter still
    uses all of them.  The colour scale is rescaled every frame -- essential
    for the likelihood visualisation, where the absolute weights are
    meaningless but their relative size is not.

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
        self.scat = ax.scatter([], [], s=4, c=[], cmap="viridis", zorder=2)
        (self.plain,) = ax.plot([], [], "b.", ms=2, zorder=2)
        self._sel = None
        self._n = None

    def set(self, X, w, colored):
        n = X.shape[1]
        if n > self.max_draw:
            if self._sel is None or self._n != n:
                self._sel = self.rng.choice(n, self.max_draw, replace=False)
                self._n = n
            X, w = X[:, self._sel], w[self._sel]
        else:
            self._sel = None
            self._n = n
        self.scat.set_visible(colored)
        self.plain.set_visible(not colored)
        if colored:
            self.scat.set_offsets(np.column_stack([X[0], X[1]]))
            self.scat.set_array(w)
            lo, hi = float(w.min()), float(w.max())
            self.scat.set_clim(lo, hi if hi > lo else lo + 1e-12)
        else:
            self.plain.set_data(X[0], X[1])

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
    return ax


class Panel:
    """Left-hand text panel: the true/model parameter table and the status."""

    def __init__(self, fig, flags=()):
        self.flags = flags
        self.text = fig.text(0.015, 0.97, "", family="monospace", fontsize=9,
                             va="top", ha="left")

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

        # Wheel r/B: model is fixed on Params (never selectable, no
        # brackets), only the true hardware value is an editable ladder
        # entry.
        rows.append("        (odometry calibration)")
        for name in FIXED_MODEL_ROWS:
            t = next(t for t in TUNABLES if t.key == ("true", name))
            sel = TUNABLES.index(t) == state.cursor
            true_txt = t.format(state.value("true", name))
            true_cell = ("[%s]" if sel else " %s ") % true_txt.center(7)
            model_cell = " %s " % t.format(getattr(params, name)).center(7)
            rows.append(f"{t.label:>7} {true_cell}{model_cell}")

        lm = " ".join(f"L{k+1}" if state.lmask[k] else " . " for k in range(len(state.lmask)))
        rows += [
            "",
            f"v = {state.tspeed:+.2f} m/s",
            f"w = {np.rad2deg(state.rspeed):+.1f} deg/s",
            f"landmarks: {lm}",
        ]
        for label, value in self.flags:
            rows.append(f"{label}: {'on' if value(state) else 'off'}")
        if extra:
            rows += ["", extra]
        rows += ["", "press 'h' for keys"]
        self.text.set_text("\n".join(rows))

    @property
    def artists(self):
        return [self.text]
