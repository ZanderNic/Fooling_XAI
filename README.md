<h1 align="center">XAI-AttackBench</h1>

<h4 align="center"><i>Black-box attacks on explanations (LIME / SHAP) under prediction constraints.</i></h4>

<p align="center">
    <img src="https://img.shields.io/badge/Black--Box%20Only-%E2%9C%85-brightgreen" alt="Black-box only" />
    <img src="https://img.shields.io/badge/Explainers-LIME%20%7C%20SHAP-orange" alt="Explainers" />
    <img src="https://img.shields.io/badge/Attacks-Hill%20Climb%20%7C%20Poisoning%20%7C%20Switching-blue" alt="Attacks" />
    <img src="https://img.shields.io/badge/Models-MLP%20%7C%20CNN1D%20%7C%20RandomForest-black" alt="Models" />
</p>

---

<p align="center">
  <b>XAI-AttackBench</b> is a modular research benchmark for evaluating the robustness of explanation methods
  against <b>black-box adversarial attacks</b>.
  The goal is to maximize <i>explanation drift</i> while keeping the <i>model prediction nearly unchanged</i>
</p>

---

## ✅ What it does

Given a configuration `(dataset, model, explainer, attack, metric)` the benchmark:

- trains a model on tabular data
- fits a model-agnostic explainer (LIME / SHAP Kernel)
- generates adversarial samples `X_adv` from `X`
- enforces prediction fidelity constraints (`<= epsilon`)
- measures explanation drift (L2/ Cosine / Spearman)
- exports results as JSON (scores, timings, counters)

**Important:** All attacks in this repository are **black-box**:
they only require access to model outputs (`predict`, `predict_proba`) and do **not** rely on gradients.

---

## 🧨 Attacks (included)

- **DistributionShiftAttack**
- **ColumnSwitchAttack**
- **DataPoisoningAttack**
- **GreedyHillClimb**

---

## 🔍 Explainers

- **LIME Tabular**
- **SHAP KernelExplainer**

---

## 📏 Metrics (explanation drift)

- **L2**
- **Cosine**
- **Spearman**

---

## 🚀 Installation (fast & simple)

Our Recommendation is to install the package into a clean virtual environment

### 1) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install the project in editable mode
```bash
pip install -e .
```
Editable mode (-e) is recommended because you can modify the source code without reinstalling the package.

After the installation is finished, you can run benchmark experiments right away.


## ▶️ Run a benchmark
To start an experiment, run the benchmark script with a specific configuration
(dataset, model, attack, explainer, metric). The script will:

- load and preprocess the dataset
- train the selected model
- fit the selected explainer
- generate adversarial samples using the chosen attack
- evaluate prediction fidelity (epsilon constraint)
- compute explanation drift scores using the selected metric
- write a results `.json` file into the `results/` directory

### Example run:
```bash
python skripts/run_benchmark.py credit RF GreedyHillClimb Lime Cosine --seed 42
```

### Arguments
The Argument format is as follows:
```bash
python skripts/run_benchmark.py <dataset> <model> <attack> <explainer> <metric> --seed <int>
```
- `<dataset>`: dataset name (e.g. `credit`, `heart_uci`, `forest`, `housing`, `prisoners`)
- `<model>`: model type (e.g. `RF`, `MLP`, `CNN1D`)
- `<attack>`: attack method (e.g. `GreedyHillClimb`, `ColumnSwitchAttack`, `DataPoisoningAttack`, `DistributionShiftAttack`)
- `<explainer>`: explanation method (e.g. `Lime`, `Shap`)
- `<metric>`: explanation drift metric (e.g. `L2`, `Cosine`, `Spearman`)
- `--seed`: random seed for reproducibility (controls model init, explainer sampling, and attack randomness). Defaults to 42
- `--num_samples`: Num samples from the test set that are used for the evaluation. Defaults to 1000

After running the command, the benchmark prints progress and saves all results as a JSON file.

## ▶️ Run a Smoketest
To just check if everything is working run a some test like:
```bash
python skripts/run_benchmark.py -s
```
When prompted either press `Enter` to select all or only select specific parts that should be included in the smoke test. It then will run all combinations of those and create a report in `results/smoketest`.

## 🔧 Extending 

This repo is designed to be extended easily:
- add new attacks via BaseAttack
- add new explainers via BaseExplainer
- add new metrics via BaseMetric
- add new datasets via BaseDataset
Most additions only require a single new file
