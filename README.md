# B-GRASP: Bayesian inference of graph weights

Code accompanying our paper on Bayesian inference of graph weights. The repository contains the forward graph models, synthetic-data generation, MAP estimation, posterior sampling, and plotting scripts used for the numerical experiments.

> **Important:** posterior samples and MAP output are written to `./stat/`. The `stat/` directory is intentionally **not included in the repository** because the sampling output is large. To reproduce posterior-based figures, first run the inference code (or place previously generated `MAP.pickle` and `samples.pickle` files in the expected `stat/` subdirectories).

## Repository structure

```text
.
├── prior.py                    # graph Matérn model / graph forward-model utilities
├── create_signal.py            # generate synthetic observations
├── MAP.py                      # MAP estimation
├── sample_posterior.py         # NUTS/MCMC posterior sampling with Pyro
├── plot_tools.py               # graph plotting utilities
│
├── paper_plots_maps_top30_hdi_v9_covid_top10.py
│                               # paper figures, including maps and posterior HDIs
├── paper_plots_transformed_hdi_v3_state_labels.py
├── paper_plots_transformed.py  # alternative/earlier paper plotting scripts
├── paper_plots.py
│
├── data/
│   └── covid_data/             # U.S. state graph and COVID-19 data
├── obs/                        # saved synthetic observations
├── plots/                      # generated paper figures
└── stat/                       # MAP estimates + MCMC samples (generated; omitted)
```

The code currently assumes that commands are run from the **repository root**, since most file paths are relative (for example `./data/...`, `./obs/...`, and `./stat/...`).

## Installation

A Python environment with the following packages is required:

- NumPy
- SciPy
- PyTorch
- Pyro (`pyro-ppl`)
- Matplotlib
- NetworkX
- pandas
- Cartopy (for the U.S. map figures)

For example:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy scipy torch pyro-ppl matplotlib networkx pandas cartopy
```

The repository does not currently pin package versions. For exact archival reproducibility, we recommend recording the versions used for a successful run, e.g.

```bash
python -V
pip freeze > requirements-lock.txt
```

## Quick start

The experiments are selected at the bottom of `create_signal.py`, `MAP.py`, and `sample_posterior.py` by commenting/uncommenting the corresponding function calls. There is currently no command-line argument interface.

A typical synthetic experiment follows this pipeline:

```text
create_signal.py  ->  obs/<experiment>/obs.pickle
        |
        v
MAP.py            ->  stat/<experiment>/MAP.pickle
        |
        v
sample_posterior.py
                  ->  stat/<experiment>/samples.pickle
        |
        v
paper plotting script
                  ->  plots/*.pdf
```

### 1. Create the `stat/` output directories

Because `stat/` is omitted from the repository, create the directories before running posterior sampling:

```bash
mkdir -p \
  stat/stationary \
  stat/stationary_linearreaction \
  stat/stationary_nonlinearreaction \
  stat/heat \
  stat/heat_linearreaction \
  stat/heat_nonlinearreaction \
  stat/stationary_linearreaction_real_covid \
  stat/stationary_nonlinearreaction_real_covid \
  stat/heat_linearreaction_real_covid \
  stat/heat_nonlinearreaction_real_covid
```

`MAP.py` creates its selected output directory automatically, but `sample_posterior.py` expects the relevant directory to exist already. Creating all of them up front avoids path errors.

### 2. Generate synthetic observations (optional)

Pre-generated observation files for the main synthetic experiments are included under `obs/`. Therefore, this step can be skipped if you only want to rerun inference using those observations.

To regenerate an observation, edit the `if __name__ == '__main__':` block at the bottom of `create_signal.py` and enable the desired function, for example:

```python
if __name__ == '__main__':
    create_signal_heat_nonlinearreaction_pytorch()
```

Then run:

```bash
python create_signal.py
```

Available synthetic generators include stationary and heat/diffusion models, with linear and nonlinear reaction variants.

### 3. Compute a MAP estimate

Edit the main block at the bottom of `MAP.py` and enable the experiment you want to run. For example, for the synthetic stationary model with a linear reaction term:

```python
if __name__ == '__main__':
    MAP_stationary_linearreaction()
```

Run:

```bash
python MAP.py
```

The result is saved as

```text
stat/<experiment>/MAP.pickle
```

The real-data variants use the normalized COVID-19 case data in `data/covid_data/y_cases_normalized.pkl`.

### 4. Draw posterior samples

Similarly, select the matching experiment in the main block of `sample_posterior.py`. For example:

```python
if __name__ == '__main__':
    sample_posterior_stationary_linearreaction()
```

Then run:

```bash
python sample_posterior.py
```

Posterior inference uses Pyro's NUTS implementation. The scripts are configured for one chain with **1000 warm-up steps and 2000 retained samples** for the main sampling routines. Depending on the experiment and hardware, this can be computationally expensive.

Samples are saved to

```text
stat/<experiment>/samples.pickle
```

The pickle contains a dictionary with the Pyro samples under the `samples` key. In the synthetic graph-weight experiments the latent graph-weight parameter is stored under `samples['x']`; time-dependent models may also contain process-noise samples.

### 5. Reproduce the paper plots

Once the required `MAP.pickle` and `samples.pickle` files exist, run the paper plotting script from the repository root:

```bash
python paper_plots_maps_top30_hdi_v9_covid_top10.py
```

This script calls the paper plotting functions and writes PDF figures to `./plots/`, including the stationary/heat, linear/nonlinear reaction, synthetic, and real-COVID experiments.

The plotting functions can also be run individually from Python. For example:

```bash
python -c "from paper_plots_maps_top30_hdi_v9_covid_top10 import plot_MAP_stationary_linearreaction; plot_MAP_stationary_linearreaction()"
```

If you only need one figure, calling its function directly avoids loading and plotting every experiment.

The other `paper_plots*.py` files contain alternative versions of the plotting code used during development. `paper_plots_maps_top30_hdi_v9_covid_top10.py` is the most complete plotting entry point in the supplied repository, including map-based graph visualizations and highest-density-interval (HDI) summaries.

## Experiments and expected `stat/` paths

| Experiment | MAP/samples directory |
|---|---|
| Stationary | `stat/stationary/` |
| Stationary + linear reaction | `stat/stationary_linearreaction/` |
| Stationary + nonlinear reaction | `stat/stationary_nonlinearreaction/` |
| Heat | `stat/heat/` |
| Heat + linear reaction | `stat/heat_linearreaction/` |
| Heat + nonlinear reaction | `stat/heat_nonlinearreaction/` |
| Stationary + linear reaction, COVID data | `stat/stationary_linearreaction_real_covid/` |
| Stationary + nonlinear reaction, COVID data | `stat/stationary_nonlinearreaction_real_covid/` |
| Heat + linear reaction, COVID data | `stat/heat_linearreaction_real_covid/` |
| Heat + nonlinear reaction, COVID data | `stat/heat_nonlinearreaction_real_covid/` |

For a given posterior-based figure, both files are normally required:

```text
stat/<experiment>/MAP.pickle
stat/<experiment>/samples.pickle
```

## COVID-19 data

The real-data experiments use a graph whose nodes are U.S. states and whose edges encode state adjacency. The relevant files are under `data/covid_data/`.

`data/covid_data/datasets.py` contains the state-adjacency construction and data preprocessing utilities. The included `us-states.csv` is the raw state-level time series used by those utilities, while `y_cases_normalized.pkl` contains the normalized case data used directly by the inference scripts.

The dataset helper code is adapted from Alexander Nikitin's `covid19-on-graphs` project; see the copyright/license notice in `data/covid_data/datasets.py`.

## Reproducibility notes

- **Run from the repository root.** Relative paths are hard-coded throughout the scripts.
- **The `stat/` directory is not distributed.** Posterior-based plots cannot be regenerated until the corresponding inference output has been produced.
- **Sampling is expensive.** The default NUTS configuration uses 1000 warm-up steps and 2000 posterior samples.
- **Experiment selection is currently manual.** Check the `__main__` block of each script before running it; the repository snapshot may have a particular experiment enabled by default.
- **Random seeds are not fixed globally.** Exact bit-for-bit reproduction should not be expected unless seeds and the software/hardware environment are controlled.
- **Cartopy may download Natural Earth map data** the first time a map-based script requests it, depending on the local Cartopy installation/cache.
- **Pickle files are Python-specific.** Only load pickle files from trusted sources.
- For a long sampling run, consider redirecting output to a log:
  ```bash
  python sample_posterior.py 2>&1 | tee sampling.log
  ```

## Suggested end-to-end reproduction workflow

For each synthetic experiment used in the paper:

```bash
# 1. Optionally regenerate the observation after selecting the function
python create_signal.py

# 2. Select the matching MAP function
python MAP.py

# 3. Select the matching posterior-sampling function
python sample_posterior.py
```

Repeat for the experiments needed for the figures. For the real COVID-19 experiments, skip synthetic signal generation and run the corresponding `_real_covid` MAP and posterior functions.

After all required `stat/` outputs have been generated:

```bash
python paper_plots_maps_top30_hdi_v9_covid_top10.py
```

The resulting PDFs are written to `plots/`.

## Citation

If you use this code, please cite the accompanying paper:

```bibtex
@article{TODO,
  title   = {TODO: paper title},
  author  = {TODO: authors},
  journal = {TODO},
  year    = {TODO}
}
```

Please replace the placeholder above with the final bibliographic information for the paper.

## License

No repository-level license file was included in the supplied snapshot. Add the intended software license here before public release. Note that individual data/helper files may carry their own attribution or license notices.
