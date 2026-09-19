#!/usr/bin/env python3
"""Checks that the ported filters do what they claim.

    python tests/test_locdemo.py

No test framework needed; it just asserts and prints.  The interesting one is
test_ekf_covariance_matches_monte_carlo: it compares the covariance the EKF
predicts against a large sample of the same motion model, which is exactly the
comparison monte_carlo_sim_odom.m was written to let you make by eye.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from locdemo import models
from locdemo.ekf import EKFLocalizer
from locdemo.params import Params, DemoState
from locdemo.pf import ParticleFilter, resample_stratified
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
