"""Run comparison or sensitivity experiments with generated samples."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from .data_generation import generate_dataset
from .metrics import mse, bias


SENSITIVITY = {
    'base': {},
    'bh16': {'base_hidden': (16,)},
    'bh48': {'base_hidden': (48,)},
    'wh8': {'weak_hidden': (8,)},
    'wh16': {'weak_hidden': (16,)},
    'eta01': {'eta': 0.1},
    'eta03': {'eta': 0.3},
    'rho01': {'rho': 0.1},
    'rho1': {'rho': 1.0},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios', nargs='+', choices=list('ABCDEF'), default=list('ABCDEF'))
    parser.add_argument('--sizes', nargs='+', type=int, default=[100, 200, 500])
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--methods', nargs='+', choices=['MonotoneEnvelope', 'FDH', 'DEA', 'C2NLS'],
                        default=['MonotoneEnvelope', 'FDH', 'DEA', 'C2NLS'])
    parser.add_argument('--sensitivity', action='store_true', help='Run the nine neural-network parameter settings.')
    parser.add_argument('--quick', action='store_true', help='Use 10 training epochs and 3 boosting rounds for an installation check.')
    parser.add_argument('--output', type=Path, default=Path('simulation/output/comparison'))
    parser.add_argument('--base-seed', type=int, default=20260101)
    parser.add_argument('--x-low', type=float, default=0.1)
    parser.add_argument('--x-high', type=float, default=10.0)
    parser.add_argument('--u-mean', type=float, default=0.0)
    parser.add_argument('--u-var', type=float, default=0.4)
    parser.add_argument('--ineff-mode', choices=['sub', 'div'], default='sub')
    parser.add_argument('--solver', default='MOSEK', help='CVXPY solver for DEA. C2NLS uses MOSEK through pyStoNED.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--base-epochs', type=int, default=2000)
    parser.add_argument('--boost-epochs', type=int, default=800)
    parser.add_argument('--boost-rounds', type=int, default=50)
    args = parser.parse_args()
    if args.trials < 1 or any(n < 1 for n in args.sizes):
        parser.error('Trial counts and sample sizes must be positive.')
    if args.x_low < 0 or args.x_high <= args.x_low or args.u_var < 0:
        parser.error('Use nonnegative ordered input bounds and a nonnegative variance.')
    if min(args.base_epochs, args.boost_epochs, args.boost_rounds) < 1:
        parser.error('Training epochs and boosting rounds must be positive.')
    if len(set(args.scenarios)) != len(args.scenarios) or len(set(args.sizes)) != len(args.sizes) or len(set(args.methods)) != len(args.methods):
        parser.error('Scenario, size and method lists must not contain duplicates.')
    if args.quick:
        args.base_epochs, args.boost_epochs, args.boost_rounds = 10, 10, 3
    if args.sensitivity:
        args.methods = ['MonotoneEnvelope']
    return args


def make_estimator(name, seed, args, override=None):
    if name == 'MonotoneEnvelope':
        from .methods.monotone_envelope import MonotoneEnvelopeEstimator, MonotoneEnvelopeParams
        params = MonotoneEnvelopeParams(seed=seed, verbose=False, device=args.device,
                                       base_epochs=args.base_epochs, boost_epochs=args.boost_epochs,
                                       T=args.boost_rounds, **(override or {}))
        return MonotoneEnvelopeEstimator(params)
    if name == 'FDH':
        from .methods.FDH_frontier import FDHEstimatorWrapper
        return FDHEstimatorWrapper(add_origin=False, fill_value=0.0)
    if name == 'DEA':
        from .methods.DEA_frontier import DEAEstimatorWrapper
        return DEAEstimatorWrapper(rts='vrs', add_origin=False, solver=args.solver)
    from .methods.C2NLS_frontier import C2NLSEstimatorWrapper
    return C2NLSEstimatorWrapper(rts='vrs', add_origin=False, solver='mosek')


def preflight(args):
    """Resolve required dependencies and solvers before writing any results."""
    for method in args.methods:
        make_estimator(method, args.base_seed, args)
    if 'DEA' in args.methods:
        import cvxpy as cp
        if args.solver not in cp.installed_solvers():
            raise RuntimeError(f'DEA solver {args.solver} is not installed. Available: {cp.installed_solvers()}')
    if 'C2NLS' in args.methods:
        import mosek  # noqa: F401 - local MOSEK installation is needed by pyStoNED


def main():
    args = parse_args()
    preflight(args)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose a new --output directory; this one already contains files: {output}')
    output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config['python_version'] = platform.python_version()
    config['source_commit'] = '06619326390d5475987ac38109153d5851a8ebd3'
    config['packages'] = {}
    for package in ['numpy', 'torch', 'cvxpy', 'pystoned', 'Mosek']:
        try:
            config['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    config['code_sha256'] = {
        p.relative_to(Path(__file__).parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.rglob('*.py'))
    }
    config['seed_rule'] = 'base_seed + (scenario_index_A0)*100000 + n*1000 + trial'
    config['sensitivity_settings'] = SENSITIVITY if args.sensitivity else {'default': {}}
    if 'MonotoneEnvelope' in args.methods:
        config['neural_defaults'] = asdict(make_estimator('MonotoneEnvelope', args.base_seed, args).params)
    (output / 'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

    fields = ['scenario', 'n', 'trial', 'method', 'robust_tag', 'seed', 'mse_in', 'bias_in',
              'elapsed_seconds', 'violations_final', 'max_gap_final', 'n_learners_final']
    logfields = ['scenario', 'n', 'trial', 'method', 'robust_tag', 'seed', 'boost_iter',
                 'max_gap', 'violations', 'n_learners_final']
    tags = SENSITIVITY if args.sensitivity else {'default': {}}
    count = 0
    with (output / 'metrics.csv').open('w', encoding='utf-8', newline='') as metrics_file, \
         (output / 'convergence.csv').open('w', encoding='utf-8', newline='') as log_file:
        writer = csv.DictWriter(metrics_file, fields)
        logger = csv.DictWriter(log_file, logfields)
        writer.writeheader()
        logger.writeheader()
        for scenario in args.scenarios:
            for n in args.sizes:
                for trial in range(args.trials):
                    seed = args.base_seed + (ord(scenario) - ord('A')) * 100000 + n * 1000 + trial
                    if not 0 <= seed < 2**32 - 1:
                        raise ValueError('The seed rule exceeded the NumPy legacy seed range.')
                    X, observed, truth, _ = generate_dataset(
                        scenario, n, x_low=args.x_low, x_high=args.x_high,
                        u_mean=args.u_mean, u_var=args.u_var, ineff_mode=args.ineff_mode, seed=seed)
                    for method in args.methods:
                        for tag, override in tags.items():
                            est = make_estimator(method, seed, args, override)
                            start = time.perf_counter()
                            est.fit(X, observed)
                            prediction = est.predict(X)
                            elapsed = time.perf_counter() - start
                            if not np.isfinite(prediction).all():
                                raise RuntimeError(f'Non-finite predictions: {scenario}/{n}/{trial}/{method}/{tag}')
                            gap = np.maximum(observed - prediction, 0)
                            learners = len(est.learners) if method == 'MonotoneEnvelope' else ''
                            common = dict(scenario=scenario, n=n, trial=trial, method=method, robust_tag=tag, seed=seed)
                            writer.writerow({**common, 'mse_in': mse(truth, prediction), 'bias_in': bias(truth, prediction),
                                             'elapsed_seconds': elapsed, 'violations_final': int((gap > 1e-6).sum()),
                                             'max_gap_final': float(gap.max()), 'n_learners_final': learners})
                            history = getattr(est, 'history', None) or {}
                            for index, (max_gap, violations) in enumerate(zip(history.get('max_gap', []), history.get('violations', [])), 1):
                                logger.writerow({**common, 'boost_iter': index, 'max_gap': max_gap,
                                                 'violations': violations, 'n_learners_final': learners})
                            metrics_file.flush()
                            log_file.flush()
                            count += 1
                            print(f'{scenario} n={n} trial={trial} {method}/{tag} MSE={mse(truth, prediction):.6g}', flush=True)
    (output / 'completed.json').write_text(json.dumps({'completed_runs': count}, indent=2), encoding='utf-8')
    print(f'Completed {count} runs. Results: {output}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ImportError, RuntimeError, ValueError, FileExistsError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
