"""2D pose-graph SLAM: batch (Gauss-Newton) optimization over a graph of
robot poses and landmark positions, both unknown, connected by noisy
odometry edges (between consecutive poses) and landmark-observation edges
(pose to landmark).

This is a different way of doing the same job as EKFSLAM: both estimate the
robot's path and a map of unknown landmarks from noisy odometry and noisy
range/bearing measurements, and both use exactly the same motion and
measurement models from models.py. The difference is *when* the estimate is
corrected. EKFSLAM folds in one measurement at a time and never revisits an
earlier pose -- by the time a loop closes, the poses along the way are long
since fixed at whatever the filter believed then. A pose graph instead keeps
every measurement as an edge and only solves the whole thing when asked to:
one loop closure can then pull the *entire* accumulated path back into shape
at once, not just wherever the robot is right now.
"""

import numpy as np

from . import models


def _accumulate(H, b, e, Omega, blocks):
    """Add one edge's contribution to the normal equations.

    `blocks` is a list of (offset, J) pairs, one per free variable this edge
    touches (a fixed variable, e.g. the anchored first pose, is simply left
    out). With e = measured - predicted and J = d(e)/d(variable), Gauss-Newton
    on sum(e^T Omega e) needs H = J^T Omega J and b = J^T Omega e, accumulated
    over every edge and every pair of blocks it touches (including a block
    against itself, for the diagonal).
    """
    for off_r, Jr in blocks:
        nr = Jr.shape[1]
        b[off_r:off_r + nr] += Jr.T @ Omega @ e
        for off_c, Jc in blocks:
            nc = Jc.shape[1]
            H[off_r:off_r + nr, off_c:off_c + nc] += Jr.T @ Omega @ Jc


class PoseGraph:
    """Growing set of pose and landmark variables, plus the edges between
    them. Nothing here is estimated until optimize() is called -- until
    then, poses/landmarks just hold whatever initial guess they were given
    (the raw, uncorrected odometry chain for poses; inverse-observation from
    the first sighting for landmarks).
    """

    def __init__(self):
        self.poses = []       # list of np.array([x, y, a]), growing
        self.landmarks = {}   # {landmark index: np.array([lx, ly])}
        self.odom_edges = []      # (i, j, delta_meas (3,), Omega (3,3))
        self.landmark_edges = []  # (pose_i, landmark_k, rho_meas, phi_meas, Omega (2,2))
        self.gps_edges = []        # (pose_i, x_meas, y_meas, Omega (2,2))

    @property
    def n_poses(self):
        return len(self.poses)

    def add_pose(self, x, y, a):
        """Append a new pose node, returning its index."""
        self.poses.append(np.array([x, y, a], dtype=float))
        return len(self.poses) - 1

    def add_odom_edge(self, i, j, delta_meas, Omega):
        self.odom_edges.append((i, j, np.asarray(delta_meas, dtype=float), Omega))

    def add_landmark_edge(self, pose_i, landmark_k, rho_meas, phi_meas, Omega, init_xy=None):
        """Add a landmark-observation edge, mapping `landmark_k` first if
        this is its first sighting (init_xy required the first time only).
        """
        if landmark_k not in self.landmarks:
            self.landmarks[landmark_k] = np.asarray(init_xy, dtype=float)
        self.landmark_edges.append((pose_i, landmark_k, rho_meas, phi_meas, Omega))

    def add_gps_edge(self, pose_i, x_meas, y_meas, Omega):
        """A direct, absolute (x, y) fix on one pose -- the only kind of
        edge here that isn't purely relative, which is exactly why it's
        useful: it's the one thing that can tie the graph to the world's
        absolute frame rather than just to its own internal consistency.
        `Omega` is deliberately *not* a near-infinite "trust this exactly"
        value the way EKFSLAM.super_gps_update's is: that filter only ever
        reconciles one fix against its current belief, one at a time, but
        a batch solver reconciles every edge at once, so two near-perfect
        fixes on different poses that disagree with the odometry between
        them would force all of that disagreement onto whatever edges
        happen to sit in between -- a real, finite Omega (the same sig_gps
        the fire-once GPS fix elsewhere already uses) lets the solver
        trade it off against everything else instead of being forced to
        honour it exactly.
        """
        self.gps_edges.append((pose_i, x_meas, y_meas, Omega))

    # ------------------------------------------------------------------
    def optimize(self, iterations=15, damping=1e-6):
        """Gauss-Newton, in place. Pose 0 is held fixed as the gauge anchor
        -- a pose graph has no absolute reference otherwise, since every
        edge only constrains *relative* poses or relative observations.
        Pinning one pose (3 DOF: x, y, heading) is exactly enough to remove
        that freedom for a rigid 2D graph, poses and landmarks alike.
        """
        if self.n_poses < 2:
            return

        landmark_keys = sorted(self.landmarks)
        pose_offset, off = {}, 0
        for i in range(1, self.n_poses):
            pose_offset[i] = off
            off += 3
        lm_offset = {}
        for k in landmark_keys:
            lm_offset[k] = off
            off += 2
        n = off
        if n == 0:
            return

        x = np.zeros(n)
        for i in range(1, self.n_poses):
            x[pose_offset[i]:pose_offset[i] + 3] = self.poses[i]
        for k in landmark_keys:
            x[lm_offset[k]:lm_offset[k] + 2] = self.landmarks[k]

        def pose_at(i):
            return self.poses[0] if i == 0 else x[pose_offset[i]:pose_offset[i] + 3]

        def lm_at(k):
            return x[lm_offset[k]:lm_offset[k] + 2]

        for _ in range(iterations):
            H = np.zeros((n, n))
            b = np.zeros(n)

            for i, j, meas, Omega in self.odom_edges:
                pi, pj = pose_at(i), pose_at(j)
                c, s = np.cos(pi[2]), np.sin(pi[2])
                dx, dy = pj[0] - pi[0], pj[1] - pi[1]
                pred = np.array([c * dx + s * dy, -s * dx + c * dy,
                                  models.wrap_angle(pj[2] - pi[2])])
                e = np.array([meas[0] - pred[0], meas[1] - pred[1],
                              models.wrap_angle(meas[2] - pred[2])])

                # d(pred)/d(pj), d(pred)/d(pi); e = meas - pred, so the
                # Jacobians of e are the negatives of these.
                dpred_dpj = np.array([[c, s, 0.0],
                                       [-s, c, 0.0],
                                       [0.0, 0.0, 1.0]])
                dpred_dpi = np.array([[-c, -s, -s * dx + c * dy],
                                       [s, -c, -(c * dx + s * dy)],
                                       [0.0, 0.0, -1.0]])

                blocks = []
                if i != 0:
                    blocks.append((pose_offset[i], -dpred_dpi))
                if j != 0:
                    blocks.append((pose_offset[j], -dpred_dpj))
                if blocks:
                    _accumulate(H, b, e, Omega, blocks)

            for pose_i, k, rho_m, phi_m, Omega in self.landmark_edges:
                p, lm = pose_at(pose_i), lm_at(k)
                rho, phi = models.range_bearing(p[0], p[1], p[2], lm[0], lm[1])
                e = np.array([rho_m - rho, models.wrap_angle(phi_m - phi)])

                # Same range/bearing Jacobians as EKF/PF/EKFSLAM use, wrt the
                # pose; the landmark's own columns are minus the pose's (x,
                # y) ones, by the same x-vs-xl symmetry EKFSLAM.update relies
                # on (range/bearing depend on (xl-x, yl-y) only).
                dpred_dpose = np.vstack([
                    models.range_jacobian(p[0], p[1], lm[0], lm[1], rho),
                    models.bearing_jacobian(p[0], p[1], lm[0], lm[1], rho),
                ])
                dpred_dlm = -dpred_dpose[:, :2]

                blocks = []
                if pose_i != 0:
                    blocks.append((pose_offset[pose_i], -dpred_dpose))
                blocks.append((lm_offset[k], -dpred_dlm))
                _accumulate(H, b, e, Omega, blocks)

            for pose_i, x_meas, y_meas, Omega in self.gps_edges:
                if pose_i == 0:
                    continue  # already exactly (x_meas, y_meas)'s best case: pinned there anyway
                p = pose_at(pose_i)
                e = np.array([x_meas - p[0], y_meas - p[1]])
                dpred_dpose = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
                _accumulate(H, b, e, Omega, [(pose_offset[pose_i], -dpred_dpose)])

            H += damping * np.eye(n)
            dx = np.linalg.solve(H, -b)
            x += dx
            for i in range(1, self.n_poses):
                x[pose_offset[i] + 2] = models.wrap_angle(x[pose_offset[i] + 2])

        for i in range(1, self.n_poses):
            self.poses[i] = x[pose_offset[i]:pose_offset[i] + 3]
        for k in landmark_keys:
            self.landmarks[k] = x[lm_offset[k]:lm_offset[k] + 2]
