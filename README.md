# Python localization demos

Demos of the Extended Kalman Filter and the Particle Filter (Monte Carlo
Localization) for localization, used during lectures.

There are four point landmarks which you can turn on and off. You control what
the robot's sensors really measure *and* what the filter believes about them,
and you drive the platform with the keyboard.

## Getting started

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python run_ekf.py        # Extended Kalman Filter
.venv/bin/python run_pf.py         # Monte Carlo Localization
.venv/bin/python run_drive.py      # odometry only, no filter
```

The figure window has to have keyboard focus for the keys to work. Press `h`
for the key list at any time.

## Keys

| | | | |
|---|---|---|---|
| `up` / `down` | v up / down | `r` | reset |
| `left` / `right` | w up / down | `u` | uniform distribution |
| `space` | stop | `d` | disturb the true pose |
| `0` | w = 0 | `i` | inject noise into P (EKF only) |
| `1`…`4` | use landmark 1…4 | `enter` | force a measurement update |
| `tab` / `shift-tab` | select a parameter | `g` | Gaussian overlay on/off |
| `>` / `<` | raise / lower it | `x` | modelled range/bearing noise off/on |
| `l` / `L` | model := true, this row / all | `c` | colour particles by weight (PF only) |
| `h` | key list | `p` | resampling on/off (PF only) |
| `q` | quit | `n` / `N` | fewer / more particles (PF only) |

`run_ekf.py` and `run_pf.py` each only wire up (and list with `h`) the keys
that do something in that filter, so the PF-only and EKF-only rows above
don't show up in the other program's key list.

Driving is set-point control: you set a speed and the robot keeps going. The
arrow keys nudge the set point up and down, so `space` is how you stop.

## True values and modelled values

The panel on the left is the heart of the demo:

```
         TRUE     MODEL
         ----     -----
 sig_td [  0  ]    0.1      motion noise, proportional to distance driven
sig_rda    0       0.1      rotation noise, proportional to the turn
 sig_rd    0       0.1      rotation noise, proportional to distance driven
sig_rho   0.1m     off      range measurement noise
sig_phi    1°      off      bearing measurement noise
```

* The **TRUE** column is what the simulated world does: how much the robot
  really slips as it drives, and how noisy its sensors really are.
* The **MODEL** column is what the filter assumes: it sets `Q` in the EKF, the
  spread of the particles in the prediction step, and `R` / the likelihood
  function in the update step. `off` in the model column means the filter does
  not use that measurement type at all.

Move the cursor with `tab`, change the selected value with `<` and `>`, and
press `l` to copy a true value into the model column (`L` for all five rows).
Press `x` to force both MODEL rows to `off` at once (and again to restore
whatever they were before) when you just want to switch to dead reckoning for
a moment.

Nothing forces the two columns to agree, and that is the point. A filter whose
model is *smaller* than the truth is over-confident: it will shrink its
covariance, stop listening to measurements and diverge. A filter whose model is
*larger* than the truth is under-confident: it survives, but throws away
precision. Try `l` on a row to see the matched case, then detune it.

## EKF

### Pure prediction

Watch how the uncertainty grows as the robot drives.

* Turn all landmarks off (`1` `2` `3` `4`) or leave both measurement rows at
  `off`, which is the default.
* Drive with the arrow keys. The ellipse is a level curve of the x,y part of
  `P`; the wedge shows ±3 sigma of the heading, the lower right element of `P`.
* Change `sig_td`, `sig_rda` and `sig_rd` in the MODEL column and see how each
  error source deforms the ellipse differently.
* Now raise the same three in the TRUE column. The true robot starts drifting
  too, and you can see whether the ellipse actually covers the error.

The prediction is `EKFLocalizer.predict` in `locdemo/ekf.py` — six lines.

### Position tracking

* Turn all four landmarks on.
* Set the MODEL `sig_rho` to `1m` with `tab` and `>`.

Drive around and see the uncertainty stay low. Then investigate:

* what a smaller or larger modelled `sig_rho` does;
* what happens when the modelled value stops matching the true one;
* bearings instead of, or together with, range.

The update is `EKFLocalizer.update` — the stacked `H`, `R` and innovation, then
the three standard lines for `K`, `X` and `P`.

### Disturbances

The EKF linearises around the current estimate: the Jacobians evaluated there,
together with `P`, decide the gain `K` and therefore how much a measurement is
allowed to move the estimate.

* All landmarks on, `sig_rho = 1m` in the model column, no bearings.
* Reset with `r` and drive a while.
* Press `d` to displace the true robot. Does the filter find its way back?
* Press `i` to inflate `P`, and try again. Why does that help?

### Global localization

Often a robot has no idea where it starts: a uniform prior. Press `u` and watch
the EKF try to represent that with a Gaussian.

## MCL — Monte Carlo Localization

Here the distribution is a set of particles, each with a state and a weight.
`c` toggles the colour coding by weight, `n` and `N` change the particle count
(100 / 1000 / 10000 / 100000 / 1000000). The filter itself is vectorised and
comfortably keeps up at a million particles; what doesn't scale is drawing
them, so the display always plots a fixed random subset (20000 by default)
however big `N` is. Which particles are in that subset is only re-rolled when
the set actually changes (a predict/update/resample), not on every redraw, so
a stationary cloud stays put on screen instead of looking like it's boiling.

### Pure prediction

Repeat the EKF experiment. The prediction is `ParticleFilter.predict`.

A particle filter can represent any distribution given enough samples, so press
`g` and compare the particle cloud against the Gaussian fitted to it. How good
is the assumption the EKF is making?

### Investigate the likelihood function

The likelihood `p(z|x)` says how likely a measurement is given a state, and the
particle filter draws it for you.

* One landmark only, range at `1m` in the model column, no bearings.
* `c` on, `N` up to 10000 particles.
* Press `u` to spread the particles over the whole space.

What shape do you expect from a single range measurement? Press `enter` once
and look. From two different range measurements? (Press `u` first to start
clean.) From a single landmark measured in both range *and* bearing?

What happens if you take two range measurements in one update and then drive
without measurements? Why?

### Pose tracking, disturbances, global localization

Repeat the EKF experiments. Note that `p` turns resampling on and off. With it
off you can watch the weights decay — the program warns you on stdout when they
approach underflow, and `sum w` and `N_eff` in the panel tell the same story.

## Questions

### Simulation and the actual filter

Which parts of the code are simulating the world and which are estimating it?
Here the answer is structural: `locdemo/world.py` is the simulation,
`locdemo/ekf.py` and `locdemo/pf.py` are the filters, and `locdemo/models.py`
holds the models they share.

### Data association

Above we always knew which landmark produced which measurement, which is
realistic if the landmarks can be told apart by appearance or by a radio
signature. How would you change these programs when they cannot?

### IMU

How would you bring an IMU into the filter?

### Other measurements

We use point landmarks. How would you change the code to use

* line segments — what does the measurement model look like?
* raw laser scans — represented how, and with what measurement model?

## Implementation notes

The filter mathematics is written out in numpy rather than taken from a
filtering library, because those fifteen lines are the thing being taught.
`locdemo/ekf.py` and `locdemo/pf.py` take `Q` and `R` explicitly, so dropping
in a library-backed implementation and checking that it agrees is a reasonable
exercise.

The particle filter is vectorised over particles and keeps the explicit loop
over landmarks, which is why 100 000 particles run at about 7 ms per step. The
weights are *not* normalised: they are multiplied by the measurement
likelihood at every update and only reset by resampling, so weight degeneracy
stays visible when resampling is switched off.

## Testing

```sh
.venv/bin/python tests/test_locdemo.py
```

The interesting check is `test_ekf_covariance_matches_monte_carlo`: it compares
the covariance the EKF predicts against 400 000 samples drawn from the same
motion model, which is the comparison `run_drive.py`'s montecarlo mode lets
you make by eye. The two agree to within 0.3%.

Every demo also runs without a window, which is useful for preparing a
lecture:

```sh
.venv/bin/python run_pf.py --headless --steps 300 --seed 1 --v 0.5 \
    --set model.rho=0.5 --set true.td=0.1 --resample
.venv/bin/python run_ekf.py --snapshot ekf.png --steps 250 --v 0.5 --w 7 \
    --set model.rho=0.5 --landmarks 1010
```
