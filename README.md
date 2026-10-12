# Production frontier estimation

Materials accompanying *Shape-constrained neural networks for flexible nonparametric production frontier estimation*.

The repository brings together the empirical application datasets and the Monte Carlo experiments used to study frontier estimation with shape-constrained neural networks.

## Contents

| Directory | Contents |
|---|---|
| [application](application/README.md) | 2024 firm-level datasets for pharmaceutical manufacturing (C27) and nonmetallic mineral products manufacturing (C30), with variable definitions and preprocessing notes. |
| [simulation](simulation/README.md) | Sample generation, frontier estimators, and command-line experiments for method comparison and sensitivity analysis. |
| [algorithm](docs/algorithm.md) | A brief description and pseudocode of the two-stage training procedure. |

## Getting started

Use Python 3.12 and install the dependencies for the experiment you want to run:

```bash
python -m pip install -r simulation/requirements.txt
```

From the repository root, run a short installation check:

```bash
python -m simulation.run_experiments --scenarios A F --sizes 20 --trials 1 --methods MonotoneEnvelope FDH --quick --output simulation/output/quick_check
```

The experiment runner generates samples from the specified scenarios and random seeds, fits the selected estimators, and writes metrics and convergence logs to the output directory. Its run record includes parameter settings, package versions and code checksums. See the [simulation guide](simulation/README.md) for the comparison grid, solver setup and sensitivity experiments.

## Empirical application

The [application datasets](application/README.md) contain 326 pharmaceutical firms and 110 nonmetallic mineral products firms in 2024. Net fixed assets and employee count are inputs; total operating revenue is the output. Monetary variables are expressed in million RMB. The application guide describes the original field names and the transformation used in the accompanying analysis.

## Citation

Please cite the associated manuscript when using these materials. Acknowledgement of CSMAR should follow the database provider's citation requirements.

## Licence

See [LICENSE](LICENSE) for the software licence. Data provenance is documented in the [application guide](application/README.md).
