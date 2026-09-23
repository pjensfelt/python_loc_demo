#!/usr/bin/env python3
"""Checks that the filters do what they claim.

    python tests/test_locdemo.py

No test framework needed; it just asserts and prints.  The interesting one is
test_ekf_covariance_matches_monte_carlo: it compares the covariance the EKF
predicts against a large sample of the same motion model, which is exactly the
comparison run_drive.py's montecarlo mode lets you make by eye.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from locdemo import models
from locdemo.ekf import EKFLocalizer
from locdemo.ekfslam import EKFSLAM
from locdemo.params import Params, DemoState
from locdemo.pf import ParticleFilter, resample_stratified
from locdemo.pgo import PoseGraph
from locdemo.world import World

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


@test
def test_wrap_angle():
    assert np.isclose(models.wrap_angle(3 * np.pi), -np.pi)
    assert np.isclose(models.wrap_angle(-3 * np.pi), -np.pi)
    assert np.isclose(models.wrap_angle(0.5 * np.pi), 0.5 * np.pi)
    assert np.isclose(models.wrap_angle(-0.5 * np.pi), -0.5 * np.pi)
    assert np.allclose(models.wrap_angle(np.array([0.1, 2 * np.pi + 0.1])), 0.1)


@test
def test_wheel_ladders_match_spec():
    """r steps in 1mm increments up to +/-2cm; B steps in 1mm increments up
    to +/-1cm, then 1cm increments up to +/-10cm."""
    from locdemo.params import _R_LADDER, _B_LADDER

    assert np.isclose(min(_R_LADDER), 0.03) and np.isclose(max(_R_LADDER), 0.07)
    assert len(_R_LADDER) == 41
    assert np.allclose(np.diff(_R_LADDER), 0.001)

    assert np.isclose(min(_B_LADDER), 0.10) and np.isclose(max(_B_LADDER), 0.30)
    assert len(_B_LADDER) == 39
    fine = [v for v in _B_LADDER if 0.19 <= v <= 0.21]
    assert np.allclose(np.diff(fine), 0.001)
    # Check each one-sided coarse run separately -- concatenating them would
    # show a spurious gap across the (excluded) fine region in the middle.
    assert np.allclose(np.diff([v for v in _B_LADDER if v <= 0.19]), 0.01)
    assert np.allclose(np.diff([v for v in _B_LADDER if v >= 0.21]), 0.01)


@test
def test_odometry_scale_matched_is_identity():
    v_scale, w_scale = models.odometry_scale(0.05, 0.2, 0.05, 0.2)
    assert v_scale == 1.0 and w_scale == 1.0


@test
def test_odometry_scale_radius_scales_both_equally():
    # A 10% too-large believed radius overestimates both v and w by 10%,
    # since both wheels' rotation-to-distance conversion is off equally.
    v_scale, w_scale = models.odometry_scale(0.05, 0.2, 0.055, 0.2)
    assert np.isclose(v_scale, 1.1)
    assert np.isclose(w_scale, 1.1)


@test
def test_odometry_scale_wheelbase_only_scales_rotation():
    # A wrong wheelbase leaves the believed forward speed alone but further
    # distorts the believed turn rate -- rotation is strictly more sensitive
    # to a wheelbase error than translation is to either error.
    v_scale, w_scale = models.odometry_scale(0.05, 0.2, 0.05, 0.25)
    assert np.isclose(v_scale, 1.0)
    assert np.isclose(w_scale, 0.2 / 0.25)


@test
def test_ekf_wheel_bias_causes_deterministic_drift():
    """A wheel radius/base mismatch must bias prediction even with zero
    noise and zero measurements -- unlike ordinary noise, it never averages
    out."""
    params = Params()
    state = DemoState()
    state.tspeed, state.rspeed = 0.5, np.deg2rad(20.0)

    world, ekf = World(params, rng=np.random.default_rng(1)), EKFLocalizer(params)
    for _ in range(50):
        world.step(state)
        ekf.predict(state)
    assert np.hypot(ekf.X[0] - world.xt, ekf.X[1] - world.yt) < 1e-9

    state.set_value("true", "r", 0.055)  # model (Params.r) is 0.05
    world2, ekf2 = World(params, rng=np.random.default_rng(1)), EKFLocalizer(params)
    for _ in range(50):
        world2.step(state)
        ekf2.predict(state)
    err = np.hypot(ekf2.X[0] - world2.xt, ekf2.X[1] - world2.yt)
    print(f"    drift from a 10% radius bias over 50 steps: {err:.3f} m")
    assert err > 0.1, err


@test
def test_link_selected_and_all_use_params_for_fixed_rows():
    params = Params()
    state = DemoState()
    state.set_value("true", "r", 0.08)
    state.set_value("true", "B", 0.35)

    state.link_all(params)
    assert state.value("true", "r") == params.r
    assert state.value("true", "B") == params.B

    state.set_value("true", "r", 0.08)
    from locdemo.params import TUNABLES
    state.cursor = TUNABLES.index(next(t for t in TUNABLES if t.key == ("true", "r")))
    state.link_selected(params)
    assert state.value("true", "r") == params.r


@test
def test_resampler_matches_weights():
    rng = np.random.default_rng(0)
    w = np.array([0.1, 0.7, 0.2])
    idx = resample_stratified(w, 200000, rng)
    got = np.bincount(idx, minlength=3) / 200000
    assert np.allclose(got, w, atol=2e-3), got
    # unnormalised weights must give the same answer
    idx = resample_stratified(w * 1e-9, 200000, rng)
    assert np.allclose(np.bincount(idx, minlength=3) / 200000, w, atol=2e-3)
    # a different output size is allowed
    assert len(resample_stratified(w, 17, rng)) == 17


@test
def test_ekf_covariance_matches_monte_carlo():
    """The EKF's predicted P must agree with sampling the same motion model."""
    params = Params()
    state = DemoState()
    state.tspeed, state.rspeed = 0.5, np.deg2rad(10.0)
    for name in ("td", "rda", "rd"):
        state.set_value("model", name, 0.1)
        state.set_value("true", name, 0.1)

    steps = 40
    ekf = EKFLocalizer(params)
    ekf.P = np.zeros((3, 3))          # start with a perfectly known pose
    for _ in range(steps):
        ekf.predict(state)

    rng = np.random.default_rng(7)
    n = 400000
    X = np.zeros((3, n))
    for _ in range(steps):
        D, DA = models.sample_motion_noise(
            state.tspeed, state.rspeed, params.dT,
            state.value("true", "td"), state.value("true", "rda"),
            state.value("true", "rd"), size=n, rng=rng)
        X[0], X[1], X[2] = models.motion_model(X[0], X[1], X[2], D, DA)

    P_mc = np.cov(X)
    sd_ekf, sd_mc = np.sqrt(np.diag(ekf.P)), np.sqrt(np.diag(P_mc))
    rel = np.abs(sd_ekf - sd_mc) / sd_mc
    print(f"    EKF sd = {np.round(sd_ekf, 4)}")
    print(f"    MC  sd = {np.round(sd_mc, 4)}  (rel. err {np.round(rel, 3)})")
    assert np.all(rel < 0.10), rel
    # the mean of the sampled poses must sit on the EKF's estimate
    assert np.allclose(X.mean(axis=1)[:2], ekf.X[:2], atol=0.02)


@test
def test_ekf_tracks_with_landmarks():
    params = Params()
    state = DemoState()
    state.tspeed, state.rspeed = 0.5, np.deg2rad(7.0)
    state.set_value("model", "rho", 0.1)
    state.set_value("true", "rho", 0.1)
    for name in ("td", "rda", "rd"):
        state.set_value("true", name, 0.1)

    rng = np.random.default_rng(3)
    world, ekf = World(params, rng=rng), EKFLocalizer(params)
    for _ in range(300):
        world.step(state)
        rho, phi = world.measure(state)
        ekf.predict(state)
        ekf.update(rho, phi, state)

    err = np.hypot(ekf.X[0] - world.xt, ekf.X[1] - world.yt)
    print(f"    position error after 300 steps: {err:.3f} m")
    assert err < 0.15, err


@test
def test_ekf_diverges_without_landmarks():
    """With no measurements the covariance must grow without bound."""
    params, state = Params(), DemoState()
    state.tspeed = 0.5
    ekf = EKFLocalizer(params)
    trace = []
    for _ in range(100):
        ekf.predict(state)
        trace.append(np.trace(ekf.P))
    assert trace[-1] > trace[0] > 0
    assert np.all(np.diff(trace) > 0)


@test
def test_pf_tracks_with_landmarks():
    params = Params()
    state = DemoState(resample=True)
    state.tspeed, state.rspeed = 0.5, np.deg2rad(7.0)
    state.set_value("model", "rho", 0.5)
    state.set_value("true", "rho", 0.1)
    for name in ("td", "rda", "rd"):
        state.set_value("model", name, 0.25)
        state.set_value("true", name, 0.1)

    rng = np.random.default_rng(5)
    world, pf = World(params, rng=rng), ParticleFilter(params, N=2000, rng=rng)
    for _ in range(300):
        world.step(state)
        rho, phi = world.measure(state)
        pf.maybe_resample(state)
        pf.predict(state)
        pf.update(rho, phi, state)

    mu = (pf.X[:2] * pf.w).sum(axis=1) / pf.w.sum()
    err = np.hypot(mu[0] - world.xt, mu[1] - world.yt)
    print(f"    position error after 300 steps: {err:.3f} m")
    assert err < 0.30, err


@test
def test_pf_global_localization():
    """From a uniform prior, range measurements must collapse the cloud."""
    params = Params()
    state = DemoState(resample=True)
    state.tspeed, state.rspeed = 0.4, np.deg2rad(15.0)
    state.set_value("model", "rho", 0.5)
    state.set_value("true", "rho", 0.1)
    for name in ("td", "rda", "rd"):
        state.set_value("model", name, 0.25)

    rng = np.random.default_rng(11)
    world, pf = World(params, rng=rng), ParticleFilter(params, N=20000, rng=rng)
    world.xt, world.yt, world.at = 5.0, 5.0, 0.0
    pf.set_uniform()
    spread0 = np.sqrt(np.cov(pf.X[:2]).trace())
    for _ in range(150):
        world.step(state)
        rho, phi = world.measure(state)
        pf.maybe_resample(state)
        pf.predict(state)
        pf.update(rho, phi, state)

    mu = (pf.X[:2] * pf.w).sum(axis=1) / pf.w.sum()
    spread = np.sqrt(np.cov(pf.X[:2]).trace())
    err = np.hypot(mu[0] - world.xt, mu[1] - world.yt)
    print(f"    spread {spread0:.2f} m -> {spread:.2f} m, error {err:.2f} m")
    assert spread < 0.2 * spread0, (spread0, spread)
    assert err < 0.5, err


@test
def test_pf_resize_keeps_distribution():
    params = Params()
    rng = np.random.default_rng(13)
    pf = ParticleFilter(params, N=1000, rng=rng)
    pf.set_uniform()
    mu_before = pf.X[:2].mean(axis=1)
    pf.set_size(10000)
    assert pf.N == 10000 and len(pf.w) == 10000
    assert np.allclose(pf.w, 1.0 / 10000)
    assert np.allclose(pf.X[:2].mean(axis=1), mu_before, atol=0.4)


@test
def test_ekfslam_noiseless_tracking_is_exact():
    """With zero noise everywhere, EKF-SLAM must reconstruct the map exactly.

    This isolates the Jacobians and state bookkeeping from calibration
    questions: any error here would be an algebra bug, not a noise-tuning
    one.
    """
    params = Params()
    state = DemoState()
    state.tspeed, state.rspeed = 0.5, np.deg2rad(10.0)
    state.set_value("model", "rho", 0.1)
    state.set_value("true", "rho", 0.0)
    state.set_value("true", "phi", 0.0)
    state.lmask[:] = True

    rng = np.random.default_rng(1)
    world, slam = World(params, rng=rng), EKFSLAM(params)
    for _ in range(300):
        world.step(state)
        rho, phi = world.measure(state)
        slam.predict(state)
        slam.update(rho, phi, state)

    err = np.hypot(slam.X[0] - world.xt, slam.X[1] - world.yt)
    assert err < 1e-9, err
    for l, mu, _ in slam.mapped_landmarks():
        true_xy = (params.xL[l], params.yL[l])
        assert np.allclose(mu, true_xy, atol=1e-9), (l, mu, true_xy)


@test
def test_ekfslam_tracks_with_landmarks():
    params = Params()
    state = DemoState()
    state.tspeed, state.rspeed = 0.5, np.deg2rad(7.0)
    for column in ("model", "true"):
        state.set_value(column, "rho", 0.1)
        state.set_value(column, "phi", np.deg2rad(1.0))
    for name in ("td", "rda", "rd"):
        state.set_value("true", name, 0.1)
    state.lmask[:] = True

    rng = np.random.default_rng(3)
    world, slam = World(params, rng=rng), EKFSLAM(params)
    for _ in range(300):
        world.step(state)
        rho, phi = world.measure(state)
        slam.predict(state)
        slam.update(rho, phi, state)

    err = np.hypot(slam.X[0] - world.xt, slam.X[1] - world.yt)
    print(f"    position error after 300 steps: {err:.3f} m")
    assert err < 0.3, err
    for l, mu, _ in slam.mapped_landmarks():
        true_xy = (params.xL[l], params.yL[l])
        assert np.hypot(mu[0] - true_xy[0], mu[1] - true_xy[1]) < 0.3, (l, mu, true_xy)


@test
def test_ekfslam_correlates_landmark_with_robot_pose():
    """A landmark's covariance must correlate with the robot pose after the
    very first fused sighting -- not just after later ones.

    H references the robot and landmark columns in the same measurement row,
    so (I - K H) mixes their covariance blocks regardless of how small that
    update's innovation happens to be; what changes with more sightings is
    how *tight* the (still correlated) estimate becomes, not whether the
    correlation exists at all.
    """
    params = Params()
    state = DemoState()
    state.set_value("model", "rho", 0.1)
    state.set_value("model", "phi", np.deg2rad(1.0))
    state.lmask[:] = False
    state.lmask[0] = True

    slam = EKFSLAM(params)
    slam.P[:3, :3] = 0.5 * np.eye(3)  # the robot is already uncertain about itself

    xl, yl = params.xL[0], params.yL[0]
    rho = np.array([np.hypot(xl - slam.X[0], yl - slam.X[1]), 0, 0, 0])
    phi = np.array([np.arctan2(yl - slam.X[1], xl - slam.X[0]) - slam.X[2], 0, 0, 0])

    slam.update(rho, phi, state)
    assert abs(slam.P[0, 3]) > 1e-3, slam.P[0, 3]


@test
def test_ekfslam_super_gps_drags_correlated_landmark():
    """A superGPS fix must move a landmark that is correlated with the robot
    pose, not just the robot itself -- that's the whole point of the demo.
    """
    params = Params()
    state = DemoState()
    state.set_value("model", "rho", 0.1)
    state.set_value("model", "phi", np.deg2rad(1.0))
    state.lmask[:] = False
    state.lmask[0] = True

    slam = EKFSLAM(params)
    slam.P[:3, :3] = 0.5 * np.eye(3)
    xl, yl = params.xL[0], params.yL[0]
    rho = np.array([np.hypot(xl - slam.X[0], yl - slam.X[1]), 0, 0, 0])
    phi = np.array([np.arctan2(yl - slam.X[1], xl - slam.X[0]) - slam.X[2], 0, 0, 0])
    slam.update(rho, phi, state)

    mu_before = slam.X[3:5].copy()
    slam.super_gps_update(slam.X[0] + 1.0, slam.X[1] + 1.0)
    moved = np.hypot(*(slam.X[3:5] - mu_before))
    print(f"    landmark moved {moved:.3f} m from a 1 m robot fix")
    assert moved > 0.05, moved


@test
def test_ekfslam_reset_clears_map():
    params = Params()
    state = DemoState()
    state.tspeed = 0.5
    state.set_value("model", "rho", 0.1)
    state.lmask[:] = True

    rng = np.random.default_rng(9)
    world, slam = World(params, rng=rng), EKFSLAM(params)
    for _ in range(20):
        world.step(state)
        rho, phi = world.measure(state)
        slam.predict(state)
        slam.update(rho, phi, state)
    assert len(slam.landmark_index) > 0

    slam.reset()
    assert slam.landmark_index == {}
    assert len(slam.X) == 3 and slam.P.shape == (3, 3)


@test
def test_disturb_is_symmetric():
    params = Params()
    rng = np.random.default_rng(17)
    offs = []
    for _ in range(4000):
        w = World(params, rng=rng)
        w.disturb()
        offs.append(w.pose)
    offs = np.array(offs)
    assert np.all(np.abs(offs.mean(axis=0)) < 0.02), offs.mean(axis=0)
    assert np.abs(offs[:, 2]).max() <= np.pi / 6 + 1e-9


def _pgo_edge_cost(g):
    """Sum of e^T Omega e over every edge -- the quantity optimize() is
    actually minimising, as opposed to true-position error (which it never
    sees and isn't guaranteed to reduce on every single noisy realisation)."""
    c = 0.0
    for i, j, meas, Omega in g.odom_edges:
        pi, pj = g.poses[i], g.poses[j]
        cth, sth = np.cos(pi[2]), np.sin(pi[2])
        dx, dy = pj[0] - pi[0], pj[1] - pi[1]
        pred = np.array([cth * dx + sth * dy, -sth * dx + cth * dy,
                          models.wrap_angle(pj[2] - pi[2])])
        e = np.array([meas[0] - pred[0], meas[1] - pred[1],
                      models.wrap_angle(meas[2] - pred[2])])
        c += e @ Omega @ e
    for pose_i, k, rho_m, phi_m, Omega in g.landmark_edges:
        p, lm = g.poses[pose_i], g.landmarks[k]
        rho, phi = models.range_bearing(p[0], p[1], p[2], lm[0], lm[1])
        e = np.array([rho_m - rho, models.wrap_angle(phi_m - phi)])
        c += e @ Omega @ e
    for pose_i, x_meas, y_meas, Omega in g.gps_edges:
        p = g.poses[pose_i]
        e = np.array([x_meas - p[0], y_meas - p[1]])
        c += e @ Omega @ e
    return c


def _pgo_synthetic_loop(rng, n_landmarks=3, n_loops=2):
    """A square driven n_loops times, noisy odometry between corners, and
    landmarks seen whenever within 3.5 units of the true pose -- including,
    for n_loops=2, a real loop closure: the far-apart poses at the start of
    each lap both see the same landmarks."""
    odom_std = np.array([0.05, 0.05, np.deg2rad(2)])
    range_std, bearing_std = 0.05, np.deg2rad(2)
    true_lms = rng.uniform(1, 4, size=(n_landmarks, 2))

    corners = [(0, 0, 0), (5, 0, 90), (5, 5, 180), (0, 5, 270)]
    true_poses = [np.array([x, y, np.deg2rad(a)]) for _ in range(n_loops) for x, y, a in corners]
    true_poses.append(true_poses[0].copy())

    g = PoseGraph()
    g.add_pose(*true_poses[0])
    noisy = [true_poses[0].copy()]
    for i in range(1, len(true_poses)):
        pi, pj = true_poses[i - 1], true_poses[i]
        c, s = np.cos(pi[2]), np.sin(pi[2])
        dx, dy = pj[0] - pi[0], pj[1] - pi[1]
        true_delta = np.array([c * dx + s * dy, -s * dx + c * dy,
                                models.wrap_angle(pj[2] - pi[2])])
        meas = true_delta + odom_std * rng.standard_normal(3)
        g.add_odom_edge(i - 1, i, meas, np.diag(1 / odom_std ** 2))
        pn = noisy[-1]
        cn, sn = np.cos(pn[2]), np.sin(pn[2])
        nx, ny = pn[0] + cn * meas[0] - sn * meas[1], pn[1] + sn * meas[0] + cn * meas[1]
        na = models.wrap_angle(pn[2] + meas[2])
        noisy.append(np.array([nx, ny, na]))
        g.add_pose(nx, ny, na)

    Omega_lm = np.diag([1 / range_std ** 2, 1 / bearing_std ** 2])
    for pose_i in range(len(true_poses)):
        p_true = true_poses[pose_i]
        for k, lm in enumerate(true_lms):
            if np.hypot(lm[0] - p_true[0], lm[1] - p_true[1]) > 3.5:
                continue
            rho, phi = models.range_bearing(p_true[0], p_true[1], p_true[2], lm[0], lm[1])
            rho_m = rho + range_std * rng.standard_normal()
            phi_m = models.wrap_angle(phi + bearing_std * rng.standard_normal())
            p_noisy = noisy[pose_i]
            init_xy = (p_noisy[0] + rho_m * np.cos(p_noisy[2] + phi_m),
                       p_noisy[1] + rho_m * np.sin(p_noisy[2] + phi_m))
            g.add_landmark_edge(pose_i, k, rho_m, phi_m, Omega_lm, init_xy=init_xy)

    return g, true_poses, true_lms


@test
def test_pgo_optimize_reduces_edge_cost():
    """Gauss-Newton must never make the thing it's minimising worse."""
    rng = np.random.default_rng(0)
    for seed in range(10):
        g, _, _ = _pgo_synthetic_loop(np.random.default_rng(seed))
        cost_before = _pgo_edge_cost(g)
        g.optimize(iterations=25)
        cost_after = _pgo_edge_cost(g)
        assert cost_after <= cost_before + 1e-6, (seed, cost_before, cost_after)


@test
def test_pgo_optimize_pins_first_pose():
    g, true_poses, _ = _pgo_synthetic_loop(np.random.default_rng(1))
    g.optimize(iterations=10)
    assert np.allclose(g.poses[0], true_poses[0])


@test
def test_pgo_optimize_recovers_loop_closure():
    """Average pose and landmark error should clearly improve over many
    noisy trials (a single trial can go either way on a weakly-constrained
    quantity, e.g. a landmark seen only twice -- see test_pgo_optimize_
    reduces_edge_cost for the guarantee that holds on every single trial)."""
    pose_err_before, pose_err_after = [], []
    lm_err_before, lm_err_after = [], []
    for seed in range(20):
        g, true_poses, true_lms = _pgo_synthetic_loop(np.random.default_rng(seed))
        pose_err_before.append(np.mean([np.hypot(g.poses[i][0] - true_poses[i][0],
                                                   g.poses[i][1] - true_poses[i][1])
                                         for i in range(len(true_poses))]))
        lm_err_before.append(np.mean([np.hypot(*(g.landmarks[k] - true_lms[k]))
                                       for k in g.landmarks]))
        g.optimize(iterations=25)
        pose_err_after.append(np.mean([np.hypot(g.poses[i][0] - true_poses[i][0],
                                                  g.poses[i][1] - true_poses[i][1])
                                        for i in range(len(true_poses))]))
        lm_err_after.append(np.mean([np.hypot(*(g.landmarks[k] - true_lms[k]))
                                      for k in g.landmarks]))

    print(f"    mean pose error {np.mean(pose_err_before):.3f} -> {np.mean(pose_err_after):.3f} m, "
          f"landmark error {np.mean(lm_err_before):.3f} -> {np.mean(lm_err_after):.3f} m")
    assert np.mean(pose_err_after) < 0.7 * np.mean(pose_err_before)
    assert np.mean(lm_err_after) < 0.85 * np.mean(lm_err_before)


@test
def test_pgo_gps_edge_reduces_cost():
    """A GPS edge is just another edge in the same sum(e^T Omega e) the
    other edge types are already checked against -- Gauss-Newton must never
    make it worse, same guarantee as test_pgo_optimize_reduces_edge_cost."""
    for seed in range(10):
        rng = np.random.default_rng(seed)
        g, true_poses, _ = _pgo_synthetic_loop(rng)
        last = g.n_poses - 1
        gps_std = 0.5
        xg = true_poses[last][0] + gps_std * rng.standard_normal()
        yg = true_poses[last][1] + gps_std * rng.standard_normal()
        g.add_gps_edge(last, xg, yg, np.diag([1 / gps_std ** 2, 1 / gps_std ** 2]))

        cost_before = _pgo_edge_cost(g)
        g.optimize(iterations=25)
        cost_after = _pgo_edge_cost(g)
        assert cost_after <= cost_before + 1e-6, (seed, cost_before, cost_after)


@test
def test_pgo_gps_edge_improves_position():
    """A single realistic (not near-infinite) GPS fix on the last pose
    should, on average, pull the whole graph closer to the true poses --
    it's the only edge type that ties the graph to the world's absolute
    frame rather than just its own internal consistency."""
    pose_err_before, pose_err_after = [], []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        g, true_poses, _ = _pgo_synthetic_loop(rng)
        last = g.n_poses - 1
        gps_std = 0.3
        xg = true_poses[last][0] + gps_std * rng.standard_normal()
        yg = true_poses[last][1] + gps_std * rng.standard_normal()
        g.add_gps_edge(last, xg, yg, np.diag([1 / gps_std ** 2, 1 / gps_std ** 2]))

        pose_err_before.append(np.mean([np.hypot(g.poses[i][0] - true_poses[i][0],
                                                   g.poses[i][1] - true_poses[i][1])
                                         for i in range(len(true_poses))]))
        g.optimize(iterations=25)
        pose_err_after.append(np.mean([np.hypot(g.poses[i][0] - true_poses[i][0],
                                                  g.poses[i][1] - true_poses[i][1])
                                        for i in range(len(true_poses))]))

    print(f"    mean pose error {np.mean(pose_err_before):.3f} -> {np.mean(pose_err_after):.3f} m (with gps edge)")
    assert np.mean(pose_err_after) < np.mean(pose_err_before)


@test
def test_pgo_gps_edge_on_pose_zero_is_noop():
    """Pose 0 is the gauge anchor and is never a free variable -- a GPS
    edge on it should be silently skipped, not crash or fight the anchor."""
    rng = np.random.default_rng(2)
    g, true_poses, _ = _pgo_synthetic_loop(rng)
    g.add_gps_edge(0, 100.0, 100.0, np.eye(2))
    pose0_before = g.poses[0].copy()
    g.optimize(iterations=10)
    assert np.array_equal(g.poses[0], pose0_before)


def main():
    failed = 0
    for fn in TESTS:
        name = fn.__name__
        try:
            fn()
            print(f"ok   {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
