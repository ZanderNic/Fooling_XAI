from data_poisoning import *

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# read the data
print("Loading data...")

DATA_PATH = r"../../data/diabetes/diabetes.csv"

df = pd.read_csv(DATA_PATH, header=0)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)


# train a classifier model
print("Training model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=3368494378,
    n_jobs=-1
)

model.fit(X_train, y_train)

print(f"Model test accuracy: {model.score(X_test, y_test):.4f}")


# prepare attack parameters
print("Preparing attack prerequisites...")

X_min, X_max = X_train.min().values, X_train.max().values
cat_mask = [False] * X_train.shape[1]
X_cat = [None] * X_train.shape[1]

# hyperparameters
N_POP = 20
N_GEN = 40
N_SAMPLE = 5

INIT_MUTATION_RATE = 1.0  # for full std evaluation
INIT_STD = 0.4

P_ELITE = 0.05
P_COMBINE = 0.1

DRIFT_THRESHOLD = 0.1
DRIFT_CONFIDENCE = 0.95

EARLY_STOPPING_PATIENCE = 5


# prepare explainer
explainer = shap.Explainer(model)

reference_explanations = produce_shap_explanations(
    explainer,
    X_train.values
)


# initialize starting population
initial_population = init_population(
    reference_data=X_train.values,
    cat_mask=cat_mask,
    population_size=N_POP,
    mutation_rate=INIT_MUTATION_RATE,
    initial_mutation_std=INIT_STD,
    individual_sample_size=N_SAMPLE,
)


# run the genetic algorithm for data poisoning attack
print("Starting evolution...")

gen_stds, gen_fitnesses, early_stopping, logging = evolve_population(
    initial_population=initial_population,
    explainer=explainer,
    reference_explanations=reference_explanations,
    drift_threshold=DRIFT_THRESHOLD,
    drift_confidence=DRIFT_CONFIDENCE,
    n_generations=N_GEN,
    elite_prop=P_ELITE,
    p_combine=P_COMBINE,
    X_data=X_train.values,
    X_min=X_min,
    X_max=X_max,
    X_cat=X_cat,
    early_stopping_patience=EARLY_STOPPING_PATIENCE
)

print("Evolution completed.")

# save results
pickle.dump(
    {
        "gen_stds": gen_stds,
        "gen_fitnesses": gen_fitnesses,
        "early_stopping": early_stopping,
        "logging": logging,
        "X_train": X_train,
        "y_train": y_train,
        "model": model,
    },
    open(f"./evolution_results_{N_POP}p_{N_SAMPLE}s_{N_GEN}g_{DRIFT_THRESHOLD}dt_{DRIFT_CONFIDENCE}dc.pkl", "wb")
)

print("Results saved.")
print("Producing evaluation plot...")

# produce an evaluation plot
eval_individual = Individual(
    data=X_train.values.copy(),
    mutation_rate=INIT_MUTATION_RATE,
    mutation_stds=early_stopping.best_stds
)
eval_individual.mutate(
    X_min=X_min,
    X_max=X_max,
    X_cat=X_cat
)

reference_predictions = model.predict(X_train.values)
mutation_predictions = model.predict(eval_individual.data)
correct_prediction = reference_predictions == mutation_predictions

global_mutation_explanations = produce_shap_explanations(
    explainer,
    eval_individual.data
)

eval_plot = plot_evaluation(
    global_reference_explanations=reference_explanations[correct_prediction].mean(axis=0),
    global_mutation_explanations=global_mutation_explanations[correct_prediction].mean(axis=0),
    standard_deviations=eval_individual.mutation_stds,
    feature_names=X_train.columns.tolist(),
    drift_threshold=DRIFT_THRESHOLD,
    drift_confidence=DRIFT_CONFIDENCE
)

eval_plot.savefig(
    f"./evaluation_plot_{N_POP}p_{N_SAMPLE}s_{N_GEN}g_{DRIFT_THRESHOLD}dt_{DRIFT_CONFIDENCE}dc.png",
    dpi=400,
    bbox_inches='tight'
)

print("Evaluation plot saved.")