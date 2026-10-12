# Monte Carlo experiments

The experiments compare the proposed neural-network frontier with FDH, DEA and corrected concave nonparametric least squares (C2NLS), and examine sensitivity to the neural-network training parameters.

## Installation

Use Python 3.12. The basic installation supports sample generation, the proposed method and FDH:

```bash
python -m pip install -r simulation/requirements.txt
```

DEA and C2NLS additionally require the baseline dependencies:

```bash
python -m pip install -r simulation/requirements-baselines.txt
```

The default baseline setup uses a local MOSEK installation and licence. DEA can use another installed CVXPY solver through `--solver`; the C2NLS interface uses MOSEK through pyStoNED.

## Method comparison

Run from the repository root:

```bash
python -m simulation.run_experiments --scenarios A B C D E F --sizes 100 200 500 --trials 50 --methods MonotoneEnvelope FDH DEA C2NLS --output simulation/output/comparison
```

For the proposed method and FDH alone:

```bash
python -m simulation.run_experiments --scenarios A B C D E F --sizes 100 200 500 --trials 50 --methods MonotoneEnvelope FDH --output simulation/output/neural_fdh
```

Inputs are sampled independently from Uniform(0.1, 10). The normal draw has mean 0 and variance 0.4. The default observation rule is y = max(0, f(x) − |u|), with u the normal draw. `--ineff-mode div` selects y = f(x)/(1 + |u|). Bounds, distribution parameters and seeds are command-line settings.

The seed for a scenario, sample size and trial is

`base_seed + scenario_index * 100000 + n * 1000 + trial`,

where A has index 0, F has index 5, trials start at 0, and base_seed defaults to 20260101. The same generated sample is used by all methods and parameter settings for that trial. Metrics are evaluated at the fitting inputs relative to the known true frontier.

## Scenarios

| Scenario | Inputs | True frontier |
|---|---:|---|
| A | 1 | 3 log(1 + x1) |
| B | 2 | 0.3 x1 + 0.3 x2 + 0.01 sqrt(x1 x2) |
| C | 3 | 0.3 x1 + 0.3 x2 + 0.3 x3 + 0.01 cbrt(x1 x2 x3) |
| D | 1 | 3 x1⁴ / (x1⁴ + 50) |
| E | 2 | 4 (x1 + x2)³ / ((x1 + x2)³ + 120) |
| F | 3 | 4 a² / (a² + 20), where a = 0.3 x1 + 0.3 x2 + 0.3 x3 + 0.01 x1 x2 x3 |

## Sensitivity analysis

```bash
python -m simulation.run_experiments --scenarios F --sizes 100 --trials 50 --sensitivity --output simulation/output/sensitivity_F
```

This runs nine settings of the proposed method. The baseline has base width 32, weak-learner width 12, step size eta = 0.2 and multiplier update rho = 0.5. Tags bh16/bh48 change base width to 16/48; wh8/wh16 change weak width to 8/16; eta01/eta03 use step size 0.1/0.3; rho01/rho1 use multiplier update 0.1/1.0.

Default training uses 2000 base-model epochs at learning rate 0.003, up to 50 boosting rounds, and 800 weak-learner epochs at learning rate 0.05. The multiplier cap is 50 and the gap tolerance is 1e-6. The default device is CPU. The runner records the effective settings in run_config.json.

## Installation check

```bash
python -m simulation.run_experiments --scenarios A F --sizes 20 --trials 1 --methods MonotoneEnvelope FDH --quick --output simulation/output/quick_check
```

`--quick` uses 10 epochs for each training stage and at most three boosting rounds. This checks the execution path; use the standard training settings for numerical evaluation. Choose a new output directory for each run.

## Output files

| File | Contents |
|---|---|
| run_config.json | Experiment settings, package versions and code checksums. |
| metrics.csv | One record per scenario, size, trial, estimator and parameter setting: MSE, signed bias, elapsed time, final gap diagnostics and final learner count. |
| convergence.csv | Pre-update outer-iteration gap checks for the proposed method. |
| completed.json | Number of completed estimator fits, written when the run finishes successfully. |

MSE is mean((prediction − true frontier)²); signed bias is mean(prediction − true frontier). Final gap diagnostics compare the returned prediction with observed output. The convergence log records checks before fitting the next weak learner, so its last row can differ from the final diagnostics when the round limit is reached.

The experiment runner generates simulated samples in memory and saves the resulting metrics and logs. The data-generating expressions and estimator mathematics follow the downloaded source version; the release adds an explicit experiment interface and run records.
