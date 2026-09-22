# Python localization demos

Demos of the Extended Kalman Filter and the Particle Filter (Monte Carlo
Localization) for localization, plus EKF-SLAM for mapping, used during
lectures.

There are four point landmarks which you can turn on and off. You control what
the robot's sensors really measure *and* what the filter believes about them,
and you drive the platform with the keyboard.

## Getting started

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python run_ekf.py        # Extended Kalman Filter (localization)
.venv/bin/python run_pf.py         # Monte Carlo Localization
.venv/bin/python run_ekfslam.py    # EKF-SLAM (mapping)
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
| `0` | w = 0 | `i` | inject noise into P (EKF/SLAM only) |
| `1`…`4` | use landmark 1…4 | `enter` | force a measurement update |
| `tab` / `shift-tab` | select a parameter | `g` | Gaussian overlay on/off |
| `>` / `<` | raise / lower it | `x` | modelled range/bearing noise off/on |
| `l` / `L` | model := true, this row / all | `c` | cycle particle colour: weight/heading/plain (PF only) |
| `h` | key list | `A` | draw all particles, not just 20000 (PF only) |
| `q` | quit | `p` | resampling on/off (PF only) |
| `t` | true robot on/off | `o` | resample once (PF only) |
| `S` | screenshot (2 PNGs, in `snapshots/`) | `n` / `N` | fewer / more particles (PF only) |
| | | `s` | superGPS fix (SLAM only) |
| | | `G` / `y` | GPS / compass fix, once (EKF/PF only) |

`run_ekf.py`, `run_pf.py` and `run_ekfslam.py` each only wire up (and list
with `h`) the keys that do something in that program, so the PF-only,
EKF-only and SLAM-only rows above don't show up in the other programs' key
lists.

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
        (fire-once sensors)
sig_gps    1m       1m      GPS fix noise (one-shot, `G`)
sig_cmp    5°       5°      compass fix noise (one-shot, `y`)
```

* The **TRUE** column is what the simulated world does: how much the robot
  really slips as it drives, and how noisy its sensors really are.
* The **MODEL** column is what the filter assumes: it sets `Q` in the EKF, the
  spread of the particles in the prediction step, and `R` / the likelihood
  function in the update step. `off` in the model column means the filter does
  not use that measurement type at all.

Move the cursor with `tab`, change the selected value with `<` and `>`, and
press `l` to copy a true value into the model column (`L` for every row).
Press `x` to force both MODEL rows to `off` at once (and again to restore
whatever they were before) when you just want to switch to dead reckoning for
a moment.

Nothing forces the two columns to agree, and that is the point. A filter whose
model is *smaller* than the truth is over-confident: it will shrink its
covariance, stop listening to measurements and diverge. A filter whose model is
*larger* than the truth is under-confident: it survives, but throws away
precision. Try `l` on a row to see the matched case, then detune it.

`sig_gps`/`sig_cmp` are different in kind from the rows above: `G`/`y` are
one-shot fixes (see below), so whether one is ever applied is already fully
controlled by *pressing the key* -- there is no separate `off` state to also
set, and the model column can never go below its smallest positive rung
(a zero model sigma is a divide-by-zero the moment a fix doesn't land exactly
on the prediction).

### GPS and compass: one-shot absolute fixes

`G` and `y` inject a single absolute position or heading measurement, once,
whenever pressed -- unlike range/bearing, which are folded into every step.
Each draws its own noisy reading from the TRUE row (plus, for the compass,
the fixed `cmp_bias` below) and fuses it using the MODEL row's sigma. Good
for demoing a single precise correction: disturb the true pose (`d`), then
press `G` and watch the estimate jump most of the way there in one step
instead of converging gradually over many landmark updates.

### Odometry calibration: a deterministic bias

Three more rows sit below the noise table:

```
        (fixed bias)
      r  0.05m    0.05m       wheel radius the odometry assumes
      B  0.20m    0.20m       wheelbase the odometry assumes
cmp_bias    0°       0°       compass hard-iron-style bias
```

These are different in kind from everything above them: `r`, `B` and
`cmp_bias` are *fixed* on the TRUE side (real hardware isn't something you
dial in mid-run) and editable only on the MODEL side, and they represent a
bias, not noise. Real odometry only ever sees wheel-encoder rotation; it has
to *assume* a wheel radius and wheelbase to convert that into distance and
turning. Get either assumption wrong and every single step is off in the
same direction — unlike the noise above, this never averages out over a long
drive, which is exactly why it can be so much more damaging in practice than
it looks like it should be. A compass is the same story: real hardware
(hard-iron interference, mounting misalignment) reads a fixed offset from
true north every time, not a fresh random error each fix.

A wrong `r` scales the odometry's believed speed *and* turn rate equally,
since both wheels' rotation-to-distance conversion is off by the same
factor. A wrong `B` scales the believed turn rate *again*, on top of that,
because the wheelbase only enters the differential (rotation) term — so a
wheelbase error is strictly more damaging to heading than a radius error is
to anything. `l`/`L` reset a detuned row back to "perfectly calibrated."

`r` steps in 1mm increments up to +/-2cm from its true 5cm — being off by
more than that on a real wheel seems unlikely. `B` steps in 1mm increments
up to +/-1cm, then in 1cm increments up to +/-10cm from its true 20cm, since
a wheelbase is easier to get grossly wrong (e.g. by mismeasuring the chassis)
than a wheel radius is.

Try it with everything else quiet: `sig_td = sig_rda = sig_rd = 0` in both
columns, landmarks off, so the *only* source of error is `r`/`B`. Detune `r`
by 10% and watch position error grow steadily with no noise at all —
compare that to how a same-sized `sig_td` looks (noisy, but centred on the
truth). Then detune `B` instead and compare how much faster heading runs
away.

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
`c` cycles the colour coding (weight / heading / plain), and `n` and `N`
change the particle count (100 / 1000 / 10000 / 100000). What doesn't scale
is drawing them, so the display always plots a fixed random subset (20000 by
default) however big `N` is -- press `A` to draw every particle instead when
a demo needs to show the whole cloud, at the cost of a slower redraw. Which
particles are in the 20000-subset is only re-rolled when the set actually
changes (a predict/update/resample), not on every redraw, so a stationary
cloud stays put on screen instead of looking like it's boiling.

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

## EKF-SLAM

Here the landmarks are *not* given to the filter: the only things it knows at
the start are its own pose and `params.xL`/`yL`, which are used solely to draw
the black ground-truth dots you're comparing against. Turning a landmark on
with `1`…`4` maps it the moment it's first seen, extending `EKFSLAM.X` and `P`
by two entries (its `x, y`) with a huge, uncorrelated initial variance —
"we have no idea where this is yet." The red circles are the estimated
landmarks; watch them shrink from invisible (variance too large to draw a
sensible ellipse) to a tight dot as they're seen again.

`EKFSLAM.predict` is the same three-line motion update as `EKFLocalizer`,
just padded with an identity block so the mapped landmarks are carried
through unchanged (and any existing correlation between them and the robot
pose survives the padding). `EKFSLAM.update` is the same stacked `H`/`R`/
innovation update, too, except a landmark's own column pair gets *minus* the
robot's `(x, y)` partials, since range and bearing only depend on the
landmark position relative to the robot.

### Mapping and re-observing a landmark

* Turn on one landmark (say `1`) and drive around a bit with `sig_rho = 0.1m`
  and `sig_phi = 1°` in the model column.
* The first sighting places the landmark exactly where that one noisy reading
  says — an ellipse you'd need a very large plot to draw. Watch it shrink
  over the next several sightings as the estimate tightens.
* Turn on a second, third and fourth landmark the same way, then drive a
  circuit that keeps re-observing all of them. `mapped = n/4` in the side
  panel tracks how many are in the state.

### Correlation between the robot and the map

* Reset, disturb the true robot (`d`), then press `i` to inflate `P` so the
  filter's own uncertainty matches the disturbance you just caused. (Skipping
  the `i` step leaves the filter overconfident about a pose it just lost —
  try it to see the difference.)
* Turn on one landmark and force several updates with `enter`. This builds up
  a strong correlation between the robot pose and that landmark: the
  covariance update mixes their blocks together on the very first fused
  sighting already, and repeated sightings sharpen it further (see the
  comment above `INIT_LANDMARK_VAR` in `locdemo/ekfslam.py` for the linear
  algebra).
* Press `s` for a superGPS fix: a near-perfect, direct reading of the robot's
  true position (see `EKFSLAM.super_gps_update`) — nothing like a real
  sensor, but a clean way to inject one precise correction and watch what
  happens to everything correlated with it. Both the robot marker *and* the
  correlated landmark jump towards their true positions together, because
  the correlation you just built up carries the correction over.

### Loop closure

* Reset, map one landmark from the start pose, then turn it off.
* Drive away and keep updating with the other landmarks off too, so
  uncertainty accumulates with no correction. Disturb the pose (`d`) and
  inject noise (`i`) to make it worse, as in the previous demo.
* Map a *different*, not-yet-seen landmark now. It inherits the robot's
  accumulated (and possibly wrong) uncertainty — and, because the robot's own
  pose estimate had drifted by the time this landmark was added, its position
  estimate is off too, in a way correlated with that drift.
* Turn that second landmark back *off*, then turn the *first* one back *on*
  and keep updating: this is loop closure. The robot pose corrects nicely,
  and the second landmark's estimate visibly moves towards its true position
  too — purely from the correlation, since it isn't being measured directly
  any more.

The correction is partial, not exact: EKF-SLAM linearises around whatever the
current estimate is, so a landmark placed while the pose was already
noticeably wrong doesn't fully snap back to the truth once the loop closes.
Try leaving the second landmark switched *on* through the loop closure
instead of turning it off — its own (biased) measurements keep confirming
its wrong position at the same time the first landmark is trying to correct
it, and the tug-of-war between the two settles on a compromise. Watching that
tension is itself the point: it's the textbook reason EKF-SLAM is described
as *inconsistent* rather than simply "noisy," and why other approaches
(pose-graph optimisation, particle-based SLAM) exist.

## Questions

### Simulation and the actual filter

Which parts of the code are simulating the world and which are estimating it?
Here the answer is structural: `locdemo/world.py` is the simulation,
`locdemo/ekf.py`, `locdemo/pf.py` and `locdemo/ekfslam.py` are the filters,
and `locdemo/models.py` holds the models they share. Note that
`locdemo/params.py`'s `xL`/`yL` are read by `world.py` (truth) and by
`ekf.py`/`pf.py` (the known map) but never by `ekfslam.py` itself, only by
`run_ekfslam.py`'s drawing code, to plot the ground truth you're comparing
the map against.

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
`locdemo/ekf.py`, `locdemo/pf.py` and `locdemo/ekfslam.py` take `Q` and `R`
explicitly, so dropping in a library-backed implementation and checking that
it agrees is a reasonable exercise.

The particle filter is vectorised over particles and keeps the explicit loop
over landmarks, which is why 100 000 particles run at about 7 ms per step. The
weights are *not* normalised: they are multiplied by the measurement
likelihood at every update and only reset by resampling, so weight degeneracy
stays visible when resampling is switched off.

`EKFSLAM`'s state grows by two entries the first time each landmark is seen,
so its `A`/`W`/`H` matrices are full-state size (built fresh every call)
rather than the fixed 3x3/3x1 of `EKFLocalizer` — deliberately so, since a
handful of landmarks keeps the state under a couple dozen entries and the
padding is what makes the robot-landmark cross-covariance fall out of the
same `A @ P @ A.T` and `(I - K H) @ P` lines already used for localization,
instead of needing separate bookkeeping for those cross terms.

The wheel radius/wheelbase bias (`models.odometry_scale`) is applied at
exactly one point in each of `EKFLocalizer.predict`, `EKFSLAM.predict`,
`ParticleFilter.predict` and `run_drive.py`'s drift sampling: it rescales
`(v, w)` into what the odometry *believes* it commanded, before that value
is turned into `D`/`DA`. `world.py`'s true trajectory never sees it — only
the filters' and dead-reckoning-estimate's belief about their own motion is
biased, which is the whole point.

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
.venv/bin/python run_ekfslam.py --snapshot ekfslam.png --steps 250 --v 0.5 --w 20 \
    --set model.rho=0.1 --set model.phi=1.0 --landmarks 1111
```
