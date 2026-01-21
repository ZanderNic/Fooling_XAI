<h1 align="center">XAI-AttackBench</h1>

<h4 align="center"><i>Black-box attacks on explanations (LIME / SHAP) under prediction constraints.</i></h4>

<p align="center">
  <img src="https://img.shields.io/badge/Black--Box%20Only-Yes-brightgreen" alt="Black-box only" />
  <img src="https://img.shields.io/badge/Explainers-LIME%20%7C%20SHAP-orange" alt="Explainers" />
  <img src="https://img.shields.io/badge/Models-MLP%20%7C%20CNN1D%20%7C%20RandomForest-black" alt="Models" />
  <img src="https://img.shields.io/badge/Attacks-RandomWalk%20%7C%20RandomWalkWithMemory%20%7C%20MonteCarlo%20%7C%20TrainLookup%20%7C%20ColumnSwitch%20%7C%20DataPoisoning%20%7C%20GreedyHillClimb-blue" alt="Attacks" />
</p>

---

<p align="center">
  <b>XAI-AttackBench</b> is a modular research benchmark for evaluating the robustness of explanation methods
  against <b>black-box adversarial attacks</b> on tabular data.
  The goal of these attacks is to maximize the <i>explanation drift</i> while keeping the
  <i>model prediction nearly unchanged</i>.
</p>

---

## :white_check_mark: What it does

Given a configuration `(dataset, model, explainer, attack, metric)` the benchmark:

- trains a model on tabular data
- fits a model-agnostic explainer (LIME / SHAP Kernel)
- generates adversarial samples `X_adv` from `X`
- enforces prediction fidelity constraints (`<= epsilon`)
- measures explanation drift (e.g. L2/ Cosine / Spearman)
- exports results as $\texttt{JSON}$ (scores, timings, counters)

> **Important:**
  All attacks in this repository are **black-box**, i.e. they only require access to model outputs (`predict`, `predict_proba`) and do **not** rely on gradients or other model internals.


## :bomb: Attacks (included)

- **RandomWalk**
- **RandomWalkWithMemory**
- **MonteCarlo**
- **TrainLookup**
- **ColumnSwitch**
- **DataPoisoning**
- **GreedyHillClimb**


## :mag: Explainers

- **LIME Tabular**
- **SHAP KernelExplainer**


## :straight_ruler: Metrics (explanation drift)

- **L2**
- **Cosine**
- **Spearman**
- **Kendall-Tau**
- **Distortion** (L1 + Kendall-Tau)


## :rocket: Installation (fast & simple)

It is recommended to install the package into a clean virtual environment:

### 1) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install the project in editable mode
```bash
pip install -e .
```
Use the editable mode (`-e`) to be able to modify the source code without reinstalling the package.

After the installation is finished, benchmark experiments can be run right away.


## :arrow_forward: Run a benchmark
To start an experiment, run the benchmark script with a specific configuration
(dataset, model, attack, explainer). The script will:

- load and preprocess the dataset
- train the selected model
- fit the selected explainer
- generate adversarial samples using the chosen attack
- evaluate prediction fidelity (epsilon constraint)
- compute explanation drift scores using the available metrics
- write the results to a $\texttt{JSON}$ file in the `results/` directory

### Arguments
The argument format is as follows:

```bash
python skripts/run_benchmark.py <dataset> <model> <attack> <explainer> --seed <int>
```

- `<dataset>`: dataset name (e.g. `credit`, `heart_uci`, `forest`, `housing`, `prisoners`)
- `<model>`: model type (e.g. `RF`, `MLP`, `CNN1D`)
- `<attack>`: attack method (e.g. `RandomWalkAttack`, `MonteCarloAttack`, `ColumnSwitchAttack`, `DataPoisoningAttack`, `GreedyHillClimb`)
- `<explainer>`: explanation method (e.g. `Lime`, `Shap`)
- `--seed`: random seed for reproducibility (controls model init, explainer sampling, and attack randomness). Defaults to $42$.
- `--num_samples`: Number of samples from the test set that are used for the evaluation. Defaults to $1,000$.
- `--smoke-test`: If set, runs a quick test over all experiment combinations.

After running the command, the benchmark prints the progress and saves all results as a $\texttt{JSON}$ file.

### Example Run
```bash
python skripts/run_benchmark.py credit RF GreedyHillClimb Lime --seed 42 --num_samples 500
```

## :arrow_forward: Run a Smoketest
To just check if everything is working, run a test like:

```bash
python skripts/run_benchmark.py -s
```

When prompted either press `Enter` to select all or only specific parts that should be included in the smoke test. Then, all of the selected combinations will be run and a report in `results/smoketest` is created.


## :wrench: Extending 

This repository is designed to be extended easily via inheritance:

- add new attacks via BaseAttack
- add new explainers via BaseExplainer
- add new metrics via BaseMetric
- add new datasets via BaseDataset

Most additions only require a single new file and a registration in `skripts/run_benchmark.py`.
