#!/usr/bin/env python3
"""Pose-graph SLAM demo.

    python run_pgo.py            open the window
    python run_pgo.py --headless --steps 400 --seed 1 --v 0.4 --w 15

Drive the robot around (same keys as every other demo); every
`pgo_node_spacing` metres of true distance travelled, a new pose node is
added to the graph, connected to the previous one by a noisy odometry edge,
plus a landmark-observation edge for every landmark within the `max_rng`
sensing-distance row -- landmark positions are *not* known, so the first
sighting of each one adds it to the graph too, the same way EKFSLAM.update
does. The magenta lines show every landmark currently in range, live, even
between node presses -- only some of those sightings turn into edges (see
`max_rng` above). Nothing is corrected as you drive; press 'O' to solve the
whole graph at once with Gauss-Newton and watch the accumulated drift snap
back into place, especially once a loop closes and revisits an old landmark.

Unlike EKF-SLAM, which folds in one measurement at a time and never
revisits an earlier pose, the pose graph keeps every measurement as an edge,
so one loop closure corrects the *entire* path, not just wherever the robot
is right now. See locdemo/pgo.py for the optimizer itself.

Press 'h' in the window (or see README.md) for the key bindings.
"""

import numpy as np
from matplotlib.collections import LineCollection

from locdemo import app, draw, keys, models
from locdemo.params import Params, DemoState
from locdemo.pgo import PoseGraph
from locdemo.world import World


def main():
    args = app.common_args(__doc__.splitlines()[0]).parse_args()
    rng = np.random.default_rng(args.seed)

    params = Params()
    state = DemoState(dispGaussApprox=False)
    # Unlike EKF/PF/EKF-SLAM, where the uncertainty ellipse still visibly
    # grows and shrinks from MODEL noise alone even with TRUE noise at its
    # global default of zero, this demo's whole point -- the solid line
    # drifting away, then snapping back on 'O' -- needs the raw odometry to
    # actually be wrong. Since pg["odom"] is now a deterministic integration
    # of commanded velocity (see step() below), that only happens with real
    # TRUE motion noise, so this one demo gets a small nonzero TRUE default
    # instead of the usual 0 -- just enough for clearly-imperfect,
    # worth-optimizing drift, not a caricature of real wheel encoders.
    for name in ("td", "rda", "rd"):
        state.set_value("true", name, 0.05)
    # A GPS fix here only ever adds an edge -- nothing moves until the next
    # 'O' -- so, unlike EKF/PF's 1m default (chosen to not swamp their live,
    # continuously-drawn fix in this ~12m-wide world), a tighter, still
    # realistic 0.1m default makes a single 'G' press actually worth
    # pressing: good enough to visibly anchor the graph without being the
    # near-infinite "super" fix add_gps_edge deliberately avoids.
    state.set_value("true", "gps", 0.1)
    state.set_value("model", "gps", 0.1)
    # Every other demo defaults model rho/phi to "off" (dead reckoning
    # first, turn landmarks on deliberately) -- but landmark edges are the
    # whole point here, so this one turns them on by default instead,
    # matching their true-column defaults. 'x' still switches to
    # odometry-only (no loop closures at all) for that contrast if wanted.
    state.set_value("model", "rho", state.value("true", "rho"))
    state.set_value("model", "phi", state.value("true", "phi"))
    # But with max_rng defaulting to infinity (see params.py), *every*
    # landmark would be in range from the very first pose -- turning
    # measurements on wouldn't demo "discover landmarks as you go", it'd
    # just dump all 4 into the graph immediately. Landmarks default off
    # (same '1'..'4' toggles as everywhere else) so driving somewhere and
    # turning one on is still a deliberate step.
    state.lmask[:] = False
    app.apply_common_args(state, args)

    world = World(params, rng=rng)

    def sense_landmarks(graph, pose_idx, gx, gy, ga):
        """Add a landmark edge (mapping it too, if this is its first
        sighting) for every landmark currently in range. Called for the
        very first pose node as well as every new one after that --
        landmark edges used to only be added when a *new* node was created,
        which meant a landmark near the starting pose could never be
        measured from there.
        """
        if not (state.use_range and state.use_bearing):
            return
        for l in range(params.NL):
            if not state.lmask[l]:
                continue
            xl, yl = params.xL[l], params.yL[l]
            dist = np.hypot(world.xt - xl, world.yt - yl)
            if dist > state.maxRange:
                continue
            rho, phi = models.range_bearing(world.xt, world.yt, world.at, xl, yl)
            rho_m = rho + state.value("true", "rho") * rng.standard_normal()
            phi_m = models.wrap_angle(phi + state.value("true", "phi") * rng.standard_normal())
            Omega_lm = np.diag([1 / state.zRhoStd ** 2, 1 / state.zPhiStd ** 2])
            init_xy = (gx + rho_m * np.cos(ga + phi_m), gy + rho_m * np.sin(ga + phi_m))
            graph.add_landmark_edge(pose_idx, l, rho_m, phi_m, Omega_lm, init_xy=init_xy)

    def fresh_graph_state():
        """Everything that gets rebuilt on 'r' -- bundled in one dict rather
        than a pile of separate `nonlocal`s, matching the list-wrapper
        pattern the other demos use to mutate closure state from step().
        """
        return dict(
            graph=PoseGraph(),
            # The running noisy odometry belief since the last node, and the
            # covariance it has accumulated over that same span (reset with
            # it) -- this is exactly EKFLocalizer.predict's own A/W/Q
            # recipe, just accumulated fresh per graph edge instead of
            # forever, since each edge is its own independent measurement of
            # "how far did we move since the last node".
            odom=np.zeros(3),
            seg_start=np.zeros(3),
            P_seg=np.zeros((3, 3)),
            dist_since_node=0.0,
            # A second, never-corrected pose chain, purely for comparison:
            # "what would you believe if you never closed the loop at all".
            pure_odom=np.zeros(3),
            pure_path=[(0.0, 0.0)],
        )

    pg = fresh_graph_state()
    pg["graph"].add_pose(0.0, 0.0, 0.0)
    sense_landmarks(pg["graph"], 0, 0.0, 0.0, 0.0)

    fig = None
    if not args.headless:
        plt = app.pyplot(args)
        keys.clear_default_keymap()
        fig = plt.figure("Pose-graph SLAM", figsize=(11, 7))
        ax = draw.setup_axes(fig, params, "Pose-graph SLAM")
        true_robot = draw.RobotArtist(ax, params, color="k")
        (pure_line,) = ax.plot([], [], "--", color="0.6", lw=1, zorder=2,
                                label="odometry only")
        (graph_line,) = ax.plot([], [], "-o", color="tab:blue", lw=1.5, ms=3, zorder=4,
                                 label="pose graph")
        sightlines = LineCollection([], colors="c", linewidths=0.6, alpha=0.5, zorder=3)
        ax.add_collection(sightlines)
        # Magenta, matching every other demo's measurement-ray colour: every
        # landmark currently in range, live, whether or not this particular
        # frame also happens to be adding a graph edge (cyan) for it -- the
        # cyan sightlines only appear at the sparse moments a *node* is
        # created, so on their own they undersell how much the robot can
        # actually see along the way.
        live_rays = draw.RayArtist(ax)
        landmarks = draw.LandmarkMapArtist(ax, color="tab:red")
        ax.legend(loc="upper left", fontsize=8)
        panel = draw.Panel(fig, flags=[("true robot", lambda s: s.showTrueRobot)])
        keys.connect(fig, state, params, ax=ax, demo="pgo", pgo=True)
        if not args.snapshot:
            print(keys.help_text(pgo=True))

    def step(_frame=0):
        # ---- simulation ------------------------------------------------
        world.step(state)
        graph = pg["graph"]

        # ---- build the graph while driving ------------------------------
        if state.moving:
            # D, DA are the *deterministic* odometry reading -- commanded
            # speed times the wheel-calibration scale, no randomness -- the
            # same convention EKFLocalizer.predict and EKFSLAM.predict use
            # for their own mean update. Real drift then only comes from
            # TRUE motion noise (the world actually not going where
            # commanded) and wheel miscalibration, never from MODEL noise:
            # that only feeds Q below, the optimizer's *belief* about how
            # uncertain this reading is, exactly like every other filter
            # here. Sampling model-noise directly into this mean (the
            # previous version of this line) meant the graph's own
            # odometry chain drifted even with TRUE noise at zero -- a real
            # bug, not the intended "hard to get accurate" difficulty.
            v_scale, w_scale = models.odometry_scale(
                state.value("true", "r"), state.value("true", "B"), params.r, params.B)
            D, DA = state.tspeed * v_scale * params.dT, state.rspeed * w_scale * params.dT
            a_prev = pg["odom"][2]
            pg["odom"] = np.array(models.motion_model(*pg["odom"], D, DA))
            A, W = models.motion_jacobians(a_prev, D)
            Q = np.diag([(D * state.value("model", "td")) ** 2,
                         (DA * state.value("model", "rda")) ** 2,
                         (D * state.value("model", "rd")) ** 2])
            pg["P_seg"] = A @ pg["P_seg"] @ A.T + W @ Q @ W.T
            pg["dist_since_node"] += abs(state.tspeed) * params.dT

        if pg["dist_since_node"] >= params.pgo_node_spacing:
            pi = pg["seg_start"]
            c, s = np.cos(pi[2]), np.sin(pi[2])
            dx, dy = pg["odom"][0] - pi[0], pg["odom"][1] - pi[1]
            delta = np.array([c * dx + s * dy, -s * dx + c * dy,
                              models.wrap_angle(pg["odom"][2] - pi[2])])
            Omega_odom = np.linalg.inv(pg["P_seg"] + 1e-9 * np.eye(3))

            prev_idx = graph.n_poses - 1
            pg_pose = graph.poses[prev_idx]
            cg, sg = np.cos(pg_pose[2]), np.sin(pg_pose[2])
            new_x = pg_pose[0] + cg * delta[0] - sg * delta[1]
            new_y = pg_pose[1] + sg * delta[0] + cg * delta[1]
            new_a = models.wrap_angle(pg_pose[2] + delta[2])
            new_idx = graph.add_pose(new_x, new_y, new_a)
            graph.add_odom_edge(prev_idx, new_idx, delta, Omega_odom)

            pp = pg["pure_odom"]
            cp, sp = np.cos(pp[2]), np.sin(pp[2])
            pg["pure_odom"] = np.array([pp[0] + cp * delta[0] - sp * delta[1],
                                         pp[1] + sp * delta[0] + cp * delta[1],
                                         models.wrap_angle(pp[2] + delta[2])])
            pg["pure_path"].append((pg["pure_odom"][0], pg["pure_odom"][1]))

            # Landmark edges need both range and bearing: a range-only or
            # bearing-only sighting can't init a 2D landmark position on its
            # own the way sense_landmarks's inverse observation model assumes.
            sense_landmarks(graph, new_idx, new_x, new_y, new_a)

            pg["seg_start"] = pg["odom"].copy()
            pg["P_seg"] = np.zeros((3, 3))
            pg["dist_since_node"] -= params.pgo_node_spacing

        # ---- one-shot requests ------------------------------------------
        if state.injectGPS:
            # Unlike EKF's gps_update, which folds the fix into the belief
            # immediately, this only adds an edge -- nothing moves until the
            # next 'O'. It anchors the *latest* node (the only one there's a
            # live "here" to fix) with a realistic zGpsStd, not a near-zero
            # variance: see PoseGraph.add_gps_edge for why a batch solver
            # can't be handed a "trust this exactly" constraint the way a
            # sequential filter can.
            xg, yg = world.measure_gps(state)
            Omega_gps = np.diag([1 / state.zGpsStd ** 2, 1 / state.zGpsStd ** 2])
            node = graph.n_poses - 1
            graph.add_gps_edge(node, xg, yg, Omega_gps)
            print(f"GPS edge added at node {node}: ({xg:.3f}, {yg:.3f}) -- press 'O' to fold it in")
            state.injectGPS = False
        if state.optimizePGO:
            graph.optimize()
            state.optimizePGO = False
        if state.reset:
            world.reset()
            pg.clear()
            pg.update(fresh_graph_state())
            pg["graph"].add_pose(0.0, 0.0, 0.0)
            sense_landmarks(pg["graph"], 0, 0.0, 0.0, 0.0)
            state.reset = False
        if state.addDisturbance:
            world.disturb()
            state.addDisturbance = False

        if fig is None:
            return []

        # ---- drawing ------------------------------------------------------
        graph = pg["graph"]
        true_robot.set_pose(*world.pose)
        true_robot.set_visible(state.showTrueRobot)
        pure_x, pure_y = zip(*pg["pure_path"])
        pure_line.set_data(pure_x, pure_y)
        gx = [p[0] for p in graph.poses]
        gy = [p[1] for p in graph.poses]
        graph_line.set_data(gx, gy)
        sightlines.set_segments([[(graph.poses[pi][0], graph.poses[pi][1]),
                                   (graph.landmarks[k][0], graph.landmarks[k][1])]
                                  for pi, k, *_ in graph.landmark_edges])
        landmarks.set([(graph.landmarks[k], np.zeros((2, 2)))
                       for k in sorted(graph.landmarks)], show_ellipse=False)

        rho_all, phi_all = models.range_bearing(world.xt, world.yt, world.at, params.xL, params.yL)
        in_range = rho_all <= state.maxRange
        live_rays.set(world.pose, rho_all, phi_all, state.lmask & in_range,
                       state.use_range and state.use_bearing)

        togo = max(0.0, params.pgo_node_spacing - pg["dist_since_node"])
        panel.update(state, params,
                     f"nodes      = {graph.n_poses}\n"
                     f"landmarks  = {len(graph.landmarks)}\n"
                     f"odom edges = {len(graph.odom_edges)}\n"
                     f"lm edges   = {len(graph.landmark_edges)}\n"
                     f"gps edges  = {len(graph.gps_edges)}\n"
                     f"next node in {togo:.2f} m")
        return (true_robot.artists + [pure_line, graph_line, sightlines] + live_rays.artists
                + landmarks.artists + panel.artists)

    app.run(fig, state, step, params.dT, args.headless, args.steps, args.snapshot)

    if args.headless:
        graph = pg["graph"]
        print(f"true      = {np.round(world.pose, 4)}")
        print(f"nodes, landmarks = {graph.n_poses}, {len(graph.landmarks)}")


if __name__ == "__main__":
    main()
