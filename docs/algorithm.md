# Two-stage frontier estimation

The estimator combines a monotone base network with nonnegative monotone weak learners. Network outputs are anchored at the origin by subtracting the network value at zero.

## Stage 1: fit the base network

Train the base network by minimising mean squared error against observed output. Its constrained weights and activation functions provide monotonicity in each input.

## Stage 2: lift the fitted frontier

At each outer iteration, evaluate positive gaps between observed output and the current frontier. If all gaps are within tolerance, stop. Otherwise, fit a nonnegative weak learner using an adjustment-size loss and a sample-wise multiplier penalty for remaining positive gaps. Add the weak learner to the frontier with step size eta.

```text
Input: observations (X, y), step size eta, tolerance eps, round limit T

Fit a monotone base network f0 to (X, y) using squared-error loss
Set f = f0

For t = 1, ..., T:
    Compute positive gaps r = max(y - f(X), 0)
    Record maximum gap and the number of gaps larger than eps
    If every gap is at most eps:
        Stop

    Initialise a nonnegative monotone weak learner h and multipliers alpha = 0
    For each weak-learner training epoch:
        Set gap = max(y - f(X) - eta * h(X), 0)
        Update h using:
            mean((eta * h(X))^2) + mean(alpha * gap)
        Update multipliers using the current epoch's gap:
            alpha = clip(alpha + rho * gap, 0, alpha_cap)

    Set f = f + eta * h

Return f and the recorded convergence checks
```

The implementation evaluates gaps again after fitting to report the final diagnostics. Reaching the iteration limit can leave positive gaps; inspect these diagnostics when choosing training settings.

For the Monte Carlo experiment, generate X and observed y under each scenario, fit all selected estimators to the same sample, and evaluate MSE and signed bias against the known true frontier. Repeat with the prescribed scenario/size/trial seeds and aggregate the metrics across trials.
