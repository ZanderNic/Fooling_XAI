import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm
import copy
import time
import os
from pathlib import Path

import json
from datetime import datetime, timezone


from typing import Literal

from xai_bench.base import BaseModel, BaseAttack, BaseExplainer, BaseDataset

# helper functions

def manhattan_distance(
    vec1: np.ndarray,
    vec2: np.ndarray
) -> float:
    """
    Computes the Manhattan distance between two vectors.

    Args:
        vec1 (np.ndarray):
            A 1D numpy array representing the first vector of explanation importance values.

        vec2 (np.ndarray):
            A 1D numpy array representing the second vector of explanation importance values.

    Returns:
        float:
            The computed Manhattan distance between the two vectors.
    """
    assert isinstance(vec1, np.ndarray)
    assert isinstance(vec2, np.ndarray)
    assert vec1.shape == vec2.shape

    return np.sum(np.abs(vec1 - vec2), axis=-1)

def kendall_tau_distance(
    vec1: np.ndarray, 
    vec2: np.ndarray
) -> float:
    """
    Computes the ranking indices of the given vectors to sort them ascensingly and
    then calculates the Kendall Tau distance between the two rankings using the
    scipy.stats.kendalltau() function.
    Since the function returns a similarity score in the range [-1, 1], it is converted
    to a distance metric in the range [0, 1], i.e. [similar, unsimilar], by computing 
    (-tau + 1) / 2.

    Args:
        vec1 (np.ndarray):
            A 2D numpy array of shape (-1, n_features) holding the first vectors of explanation
            importance values with shape (n_features,) each.

        vec2 (np.ndarray):
            A 2D numpy array of shape (-1, n_features) holding the second vectors of explanation
            importance values with shape (n_features,) each.

    Returns:
        float:
            The normalized Kendall Tau distance between the two vectors.
    """
    assert isinstance(vec1, np.ndarray)
    assert isinstance(vec2, np.ndarray)
    assert vec1.ndim == 2
    assert vec1.shape == vec2.shape

    # produce rankings
    vec1_ranking = np.argsort(abs(vec1), axis=-1, stable=True)
    vec2_ranking = np.argsort(abs(vec2), axis=-1, stable=True)

    # compute kendall tau distance for each instance
    taus = np.array([
        kendalltau(ranking_1, ranking_2)[0]
        for ranking_1, ranking_2 in zip(vec1_ranking, vec2_ranking)             
    ])

    return (-taus + 1) / 2

def manipulation_loss(
    explanation_importance_values: tuple[np.ndarray, np.ndarray],
    distance_fn = None,
    order_ranking_fn = None,
    weights = (0.5, 0.5)
) -> float:
    """
    Produces a loss value based on a weighted combination of a distance function and an order ranking
    function. The functions are applied to the explanation importance values from an explainability 
    method evaluating a data instance before and after a manipulation.

    Args:
        explanation_importance_values (tuple):
            A tuple containing two elements:
            - The importance values before the manipulation attack for every data feature.
            - The importance values after the manipulation attack for every data feature.

        distance_fn (callable):
            A function that computes the distance between two vectors of explanation importance values,
            e.g., L1 or L2 distance. The distance value is transformed to a [0, 1] range before being 
            used in the final loss computation. 1 corresponds to maximum distance.

        order_ranking_fn (callable):
            A function that computes a ranking-based metric, e.g. Kendall Tau distance. This metric 
            should produce values in the interval [0;1].

        weights (tuple):
            A tuple containing two weights (w1, w2) that determine the contribution of the distance
            function and the order ranking function to the final loss value respectively.
    
    Returns:
        float:
            The computed manipulation loss value.
    """
    assert isinstance(explanation_importance_values, tuple)
    assert len(explanation_importance_values) == 2
    assert explanation_importance_values[0] is not None
    assert explanation_importance_values[1] is not None
    assert distance_fn is not None
    assert order_ranking_fn is not None
    assert len(weights) == 2
    assert all(isinstance(w, (int, float)) and 0 <= w for w in weights)

    # produce distance metrics in [0, infinity)
    explanation_distance = distance_fn(
        explanation_importance_values[0],
        explanation_importance_values[1]
    )

    # transform the distance to a [0, 1] range
    # with the assumption distance -> infinity => loss -> 1
    distance_transform = lambda x: 1 / (-x - 1) + 1
    explanation_distance = distance_transform(explanation_distance)

    # produce order ranking distance in [0, 1]
    order_ranking_distance = order_ranking_fn(
        explanation_importance_values[0],
        explanation_importance_values[1]
    )

    return weights[0] * explanation_distance + \
           weights[1] * order_ranking_distance

def explanation_drift(
    mutation_explanations: np.ndarray,
    reference_explanations: np.ndarray,
) -> np.ndarray:
    """
    Computes the explanation drift measure for a mutated individual based on the 
    manipulation loss between the explanation importance values before and after the mutation.
    The used distance function is the Manhattan distance and the order ranking function
    is the Kendall Tau distance.
    An explanation drift score close to 1 indicates that the mutation has significantly altered
    the explanations compared to the reference individual. A score close to 0 indicates that the
    explanations remain similar to the reference individual.

    Args:
        mutation_explanations (np.ndarray):
            A 2D numpy array of shape (n_samples, n_features) containing the explanation
            importance values for the mutated individual.

        reference_explanations (np.ndarray):
            A 2D numpy array of shape (n_samples, n_features) containing the explanation
            importance values for the reference individual.

    Returns:
        np.ndarray:
            A 1D numpy array of shape (n_samples,) containing the computed drift scores.
    """
    loss = manipulation_loss(
        explanation_importance_values = (
            mutation_explanations,
            reference_explanations
        ),
        distance_fn = manhattan_distance,
        order_ranking_fn = kendall_tau_distance,
        weights = (0.5, 0.5)
    )

    return loss

def LCB_Wilson(
    estimated_probability: np.ndarray,
    sample_size: np.ndarray,
    confidence: float = 0.95
) -> np.ndarray:
    """
    Computes the lower confidence bound using the Wilson score interval.

    Args:
        estimated_probability (np.ndarray):
            A 1D numpy array of shape (n_pop,) containing the estimated probabilities.

        sample_size (int):
            The total number of samples.

        confidence (float):
            The desired confidence level (between 0 and 1).

    Returns:
        np.ndarray:
            A 1D numpy array of shape (n_pop,) containing the computed lower confidence bounds.
    """
    assert isinstance(estimated_probability, np.ndarray)
    assert estimated_probability.ndim == 1
    assert isinstance(sample_size, np.ndarray)
    assert sample_size.ndim == 1
    assert estimated_probability.shape == sample_size.shape
    assert np.all(sample_size > 0)
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1

    # standard normal confidence interval z_{alpha/2}
    z = norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2 / sample_size
    adjusted_prob = estimated_probability + z**2 / (2 * sample_size)
    lcb = (
        adjusted_prob - \
        (z / (2 * sample_size)) * \
        np.sqrt(
            4 * sample_size * estimated_probability * \
            ( 1 - estimated_probability ) + z**2
        )
    ) / denominator

    return lcb

class Individual:
    """
    A class representing an individual of the evolutionary algorithm storing the standard
    deviations for each feature mutation, the mutation rate, and the data instances.
    """
    def __init__(
        self,
        data: np.ndarray,
        mutation_rate: float,
        mutation_stds: np.ndarray
    ):
        self.data = data
        self.mutation_rate = mutation_rate
        self.mutation_stds = mutation_stds

    def mutate(
        self,
        X_min: np.ndarray,
        X_max: np.ndarray,
        X_cat: list,
        rng: np.random.Generator
    ):
        """
        Mutates the individual data instance by applying N(0, mutation_stds[i]) gaussian white
        noise mutations to each continuous feature and another categorical value to each 
        categorical feature with the individuals given mutation probability.
        The mutated data is clipped to the provided feature bounds.

        Args:
            X_min (np.ndarray):
                A 1D numpy array representing the minimum bounds for each feature.

            X_max (np.ndarray):
                A 1D numpy array representing the maximum bounds for each feature.

            X_cat (list):
                A list of numpy arrays representing the possible categories for a categorical
                feature or None if the feature is continuous. The length of the list should be
                equal to the number of features in the individual.
            
            rng (np.random.Generator):
                A numpy random Generator instance for reproducibility.
        """
        assert isinstance(X_min, np.ndarray)
        assert isinstance(X_max, np.ndarray)
        assert X_min.ndim == 1
        assert X_max.ndim == 1
        assert X_min.shape == X_max.shape
        assert X_min.shape[0] == self.data.shape[1]
        assert isinstance(X_cat, list)
        assert len(X_cat) == self.data.shape[1]
        assert all(isinstance(cat, np.ndarray) or cat is None for cat in X_cat)
        assert isinstance(rng, np.random.Generator)

        # generate white noise mutation with feature std
        white_noise = rng.normal(
            0,
            self.mutation_stds,
            size=self.data.shape
        )

        # decide which instances to mutate
        mutation_mask = rng.random(self.data.shape[0]) < self.mutation_rate            

        # create gaussian mutations with respective stds
        for feature in range(self.data.shape[1]):
            # categorical feature mutation
            if X_cat[feature] is not None:
                # map each continuous mutation value to the nearest category value
                indices = np.abs(
                    self.data[:, feature, None] + \
                    white_noise[:, feature, None] - \
                    X_cat[feature][None, :]
                ).argmin(axis=-1)
                mutations = X_cat[feature][indices]

                # apply mutations according to the mask
                self.data[mutation_mask, feature] = mutations[mutation_mask]
            # continuous feature mutation
            else:
                # apply mutations according to the mask
                self.data[mutation_mask, feature] += white_noise[mutation_mask, feature]

        # clip the mutated data to the feature bounds
        self.data = np.clip(
            self.data,
            X_min,
            X_max
        )

def init_population(
    reference_data: np.ndarray,
    cat_mask: list = None,
    population_size: int = 50,
    mutation_rate: float = 0.1,
    initial_mutation_std: float = 0.3,
    individual_sample_size: int = 15,
    rng: np.random.Generator = None
) -> list[list[Individual]]:
    """
    Initilizes a population of n individuals by mutating copies of the provided reference data.

    Args:
        reference_data (np.ndarray):
            A numpy array of shape (n_samples, n_features) representing the reference data
            for the individuals to be mutated for the population initialization.

        cat_mask (list):
            A list of booleans indicating whether a feature is categorical (True) or continuous (False).

        n_population (int):
            The number of individuals to initialize the population with.

        initial_mutation_std (float):
            The initial mutation standard deviation to use for the individuals in the population.
            This rate is used to control the amount of mutation applied to each individual's data during
            initialization.

        rng (np.random.Generator):
            A numpy random Generator instance for reproducibility.

    Returns:
        list[list[Individual]]:
            A population of lists of Individual objects, where each list corresponds to an individual
            containing multiple samples. Each Individual has data of shape (n_samples, n_features) and 
            is mutated with the same mutation rate and mutation standard deviation.
    """
    assert isinstance(reference_data, np.ndarray)
    assert isinstance(cat_mask, list)
    assert len(cat_mask) == reference_data.shape[1]
    assert all(isinstance(is_cat, bool) for is_cat in cat_mask)
    assert isinstance(population_size, int)
    assert population_size > 0
    assert isinstance(mutation_rate, float)
    assert 0 < mutation_rate <= 1
    assert isinstance(initial_mutation_std, float)
    assert 0 <= initial_mutation_std <= 1
    assert isinstance(individual_sample_size, int)
    assert individual_sample_size > 0
    assert isinstance(rng, np.random.Generator)

    population = []

    # determine random mutation stds for each individual
    mutation_stds = np.abs(rng.normal(
        loc=initial_mutation_std,
        scale=0.015,
        size=(population_size, reference_data.shape[1])
    ))

    for i in range(population_size):
        individual_list = []

        for _ in range(individual_sample_size):
            individual_data = reference_data.copy()
            individual = Individual(
                data = individual_data,
                mutation_rate = mutation_rate,
                mutation_stds = mutation_stds[i]
            )
            individual.mutate(
                X_min = reference_data.min(axis=0),
                X_max = reference_data.max(axis=0),
                X_cat = [
                    np.unique(reference_data[:, feature]) if (
                        cat_mask[feature]
                    ) else (
                        None
                    ) for feature in range(reference_data.shape[1])
                ],
                rng = rng
            )

            individual_list.append(individual)

        population.append(individual_list)

    return population

def estimate_probability(
    drift_scores: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float = 0.3
) -> np.ndarray:
    """
    Estimates the probability that the drift scores exceed the given threshold for all valid
    entries in the drift score array identified by the valid_mask.

    Args:
        drift_scores (np.ndarray):
            A 2D numpy array of shape (n_population, n_samples) containing the globalized drift score
            for each sample of mutation standard deviations in the population. 

        valid_mask (np.ndarray):
            A 2D numpy array of shape (n_population, n_samples) containing boolean values that indicate
            whether the corresponding drift score entry is valid (True) or invalid (False).

        threshold (float):
            A float value representing the threshold to compare the drift scores against to evaluate
            Probability(drift_score >= threshold).

    Returns:
        np.ndarray:
            A 1D numpy array of shape (n_population,) containing the estimated probabilities
            for each individual in the population. 
    """
    assert isinstance(drift_scores, np.ndarray)
    assert drift_scores.ndim == 2
    assert isinstance(valid_mask, np.ndarray)
    assert all(isinstance(mask_entry, (bool, np.bool_)) for mask_entry in valid_mask.flatten())
    assert valid_mask.shape == drift_scores.shape
    assert isinstance(threshold, float)
    assert 0 < threshold

    # amount valid drift score entries
    valid_drift_scores_amount = np.sum(valid_mask, axis=-1)

    # cummulate the amount of valid drift scores that exceed the threshold
    exceeding_drift_scores_amount = np.sum(
        (drift_scores * valid_mask) >= threshold,
        axis=-1
    )

    # compute the estimated probabilities
    return exceeding_drift_scores_amount / valid_drift_scores_amount

def population_fitness(
    population: list[list[Individual]],
    explainer: BaseExplainer,
    reference_explanations: np.ndarray,
    model: BaseModel,
    reference_predictions: np.ndarray,
    drift_threshold: float = 0.5,
    drift_confidence: float = 0.95,
    prediction_threshold: float = None
) -> tuple[np.ndarray, float]:
    """
    Computes the optimization metrics for a population of individuals. The metrics are based on 
    the individuals' explanation drift scores compared to reference explanations. The values are 
    determined as tuples of (||stds||, violation_measure) where ||stds|| is the 1-norm of the mutation
    standard deviations and violation_measure is the amount by which the lower confidence bound of
    the estimated probability, that the drift scores exceed the given threshold, violates the drift
    confidence constraint defined by drift_confidence.

    Args:
        population (list[list[Individual]]):
            A population of individuals represented as a list of samples of Individual objects where
            each sample share the same underlying standard deviation for mutating the individuals' data.

        explainer (BaseExplainer):
            An explainer object that can compute feature importance values for instances.

        reference_explanations (np.ndarray):
            A 2D numpy array of shape (n_datapoints, n_features) containing the reference explanation
            importance values for each data feature.

        drift_threshold (float):
            A float value representing the threshold to compare the drift scores against to evaluate
            Probability(drift_score >= threshold).

        drift_confidence (float):
            A float value representing the desired confidence level (between 0 and 1) for the
            drift constraint, such that LCB_Wilson(estimated_probability) >= drift_confidence.

        prediction_threshold (float):
            A float value representing the threshold for model predictions to be considered valid.

    Returns:
        tuple[np.ndarray, float]:
            - A 2D numpy array of shape (n_population, 2) with the optimization metrics for each
              individual aggregated over the samples as tuples of (||stds||, violation_measure).
            - The mean estimated probability over the entire population.
    """
    assert isinstance(population, list)
    assert all(isinstance(individual_list, list) for individual_list in population)
    assert all(isinstance(individual, Individual) for individual_list in population for individual in individual_list)
    assert all(individual.data.ndim == 2 for individual_list in population for individual in individual_list)
    assert isinstance(explainer, BaseExplainer)
    assert isinstance(reference_explanations, np.ndarray)
    assert reference_explanations.ndim == 2
    assert isinstance(model, BaseModel)
    assert isinstance(reference_predictions, np.ndarray)
    assert reference_predictions.ndim == 2
    assert isinstance(drift_threshold, float)
    assert 0 <= drift_threshold
    assert isinstance(drift_confidence, float)
    assert 0 <= drift_confidence <= 1

    # determine dimensionalities
    n_pop = len(population)
    n_sample_size = len(population[0])
    n_datapoints = reference_explanations.shape[0]

    print("Start population fitness calculation:")
    start_time = time.time()  # LOGGING

    # unify all individuals' data in a single array without adding a new axis
    pop = np.concatenate(
        [individual.data for individual_list in population for individual in individual_list],
        axis=0
    )

    print(f"\tConcatenating population took {time.time() - start_time} seconds.")
    section_time = time.time()  # LOGGING

    # produce SHAP explanations for the entire population
    explanations = explainer.explain_parallel(
        pop,
        n_workers=os.cpu_count(),
        batch_size=48
    )

    print(f"\tProducing explanations took {time.time() - section_time} seconds.")
    section_time = time.time()  # LOGGING

    # repeat the reference explanations for all individuals
    # to match the above explanations shape
    reference_explanations_rep = np.tile(
        reference_explanations,
        (n_pop, n_sample_size, 1, 1)
    ).reshape(-1, reference_explanations.shape[1])

    print(f"\tProducing reference explanations took {time.time() - section_time} seconds.")
    section_time = time.time()  # LOGGING

    # calculate the drift scores for all samples
    drift_scores = explanation_drift(
        mutation_explanations = explanations,
        reference_explanations = reference_explanations_rep
    ).reshape(n_pop, -1)

    print(f"\tCalculating drift scores took {time.time() - section_time} seconds.")
    section_time = time.time()  # LOGGING

    # determine the prediction mask to filter valid explanations
    reference_predictions_rep = np.tile(
        reference_predictions,
        (n_pop, n_sample_size, 1, 1)
    )

    predictions = model.predict_raw(pop).reshape(n_pop, n_sample_size, n_datapoints, -1)
    prediction_mask = np.abs(reference_predictions_rep - predictions) <= prediction_threshold
    prediction_mask = prediction_mask.all(axis=-1).reshape(n_pop, -1)

    valid_scores_amount = prediction_mask.sum(axis=-1)

    print(f"\tCalculating prediction mask took {time.time() - section_time} seconds.")
    section_time = time.time()  # LOGGING

    # estimate the probabilities that the drift scores exceed the given threshold
    estimated_probabilities = estimate_probability(
        drift_scores = drift_scores,
        valid_mask = prediction_mask,
        threshold = drift_threshold
    )
    print(f"\tEstimated probabilities:\n{estimated_probabilities}")

    print(f"\tCalculating estimated probs took {time.time() - section_time} seconds.")
    section_time = time.time()  # LOGGING

    # determine the lower confidence interval bound for the estimated probabilities
    lcb = LCB_Wilson(
        estimated_probability = estimated_probabilities,
        sample_size = valid_scores_amount,
        confidence=0.95
    )
    print(f"\tLCB values:\n{lcb}")

    mean_probability = lcb.mean()
    print(f"\tLCB mean: {mean_probability}")

    print(f"\tCalculating LCB took {time.time() - section_time} seconds.")
    section_time = time.time()  # LOGGING

    # determine which individuals violate the drift constraint
    violation = drift_confidence - lcb

    # collect population stds and calculate their 1-norms
    stds = np.array([individual_list[0].mutation_stds for individual_list in population])
    stds = np.abs(stds).sum(axis=1)

    # determine fitnesses as tuples of (||stds||, violation_measure)
    fitnesses = np.array(list(zip(stds, violation)))

    print(f"Calculating optimization metrics took {time.time() - start_time} seconds.")

    return fitnesses, mean_probability

def rank_individual_stds(
    population: list[list[Individual]],
    fitnesses: np.ndarray
) -> np.ndarray:
    """
    Ranks the individuals' standard deviations according to their fitnesses. The ranking is done
    primarily by the violation componnt of the fitnesses ascendingly and secondarily by the
    1-norm of the individuals' standard deviations ascendingly.

    Args:
        population (list[list[Individual]]):
            A population of individuals represented as a list of samples of Individual objects where
            each sample share the same underlying standard deviation for mutating the individuals' data.

        fitnesses (np.ndarray):
            A 2D numpy array of shape (n_population, 2) containing the optimization metrics for each
            individual as tuples of (||stds||, violation_measure).

    Returns:
        np.ndarray:
            A 2D numpy array of shape (n_population, 2 + n_features) containing the ranked optimization
            metrics for each individual as tuples of (||stds||, violation_measure, std_1, std_2, ..., std_n).
    """
    assert isinstance(population, list)
    assert all(isinstance(individual_list, list) for individual_list in population)
    assert all(isinstance(individual, Individual) for individual_list in population for individual in individual_list)
    assert isinstance(fitnesses, np.ndarray)
    assert fitnesses.ndim == 2
    assert fitnesses.shape[0] == len(population)

    # collect population stds
    stds = np.array([individual_list[0].mutation_stds for individual_list in population])

    # sort fitnesses primarily by the second column (violation) ascendingly,
    # secondarily by the first column (||stds||) ascendingly
    sorted_idx = np.lexsort((fitnesses[:, 0], fitnesses[:, 1]))
    sorted_fitnesses = fitnesses[sorted_idx]

    # sort stds according to the fitness sorting
    sorted_stds = stds[sorted_idx]

    # add the stds to the fitnesses
    ranked_fitnesses = np.concatenate(
        (sorted_fitnesses, sorted_stds),
        axis=1
    )

    return ranked_fitnesses

def select_elitists(
    ranked_metrics: np.ndarray,
    elite_prop: float = 0.05
) -> list[np.ndarray]:
    """
    Selects the top `elite_prop` proportion of individuals from the population based on their
    fitness score and returns a list of their standard deviations.

    Args:
        ranked_metrics (np.ndarray):
            A 2D numpy array of shape (n_population, 2 + n_features) containing the ranked optimization
            metrics for each individual as tuples of (||stds||, violation_measure, std_1, std_2, ..., std_n).

        elite_prop (float):
            A float value between 0 and 1 representing the proportion of individuals to select as elitists.

    Returns:
        list[np.ndarray]:
            A list of the selected elite individuals' standard deviations.
    """
    assert isinstance(ranked_metrics, np.ndarray)
    assert ranked_metrics.ndim == 2
    assert isinstance(elite_prop, float)
    assert 0 < elite_prop < 1

    # determine the number of elite individuals to select
    n_elite = max(1, int(elite_prop * ranked_metrics.shape[0]))

    # select the indices of the top n_elite individuals
    elite_individual_metrics = ranked_metrics[:n_elite]

    return [entry[2:].copy() for entry in elite_individual_metrics]

def select_parent_indices(
    ranked_metrics: np.ndarray,
    n_pairs: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Selects parent indices for crossover from the individuals in the population
    based on their fitness values.
    The probability of selecting an individual is antiproportional to its fitness value
    using a softmax distribution, i.e. better individuals have a higher chance of being selected.

    Args:
        ranked_metrics (np.ndarray):
            A 2D numpy array of shape (n_population, 2 + n_features) containing the ranked optimization
            metrics for each individual as tuples of (||stds||, violation_measure, std_1, std_2, ..., std_n).

        n_pairs (int):
            The number of parent pairs to select.

        rng (np.random.Generator):
            A numpy random Generator instance for reproducibility.

    Returns:
        np.ndarray:
            A 2D numpy array of shape (n_population, 2) containing pairs of selected parent indices
            for crossover.
    """
    assert isinstance(ranked_metrics, np.ndarray)
    assert ranked_metrics.ndim == 2
    assert isinstance(n_pairs, int)
    assert n_pairs > 0
    assert isinstance(rng, np.random.Generator)

    # softmax weights for all individuals
    softmax = lambda x: np.exp(x) / np.sum(np.exp(x))
    # Since lower fitness values are better, the values are
    # negated to give higher weights to better individuals
    individual_weights = softmax(-1 * (ranked_metrics[:, :2].sum(axis=1)))

    # determine parent pairs from all individuals
    parent_pair_indices = rng.choice(
        np.arange(ranked_metrics.shape[0]),
        size=(n_pairs, 2),
        replace=True,
        p=individual_weights
    )

    return parent_pair_indices

def crossover_parent_stds(
    dominant_fitness_metrics: np.ndarray,
    recessive_fitness_metrics: np.ndarray,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Performs crossover of standard deviations between the given parent individuals to produce
    offspring standard deviations with values inherited from both parents based on their
    fitness metrics.

    Args:
        dominant_fitness_metrics (np.ndarray):
            A numpy array of the form (||stds||, violation_measure, std_1, std_2, ..., std_n).

        recessive_fitness_metrics (np.ndarray):
            A numpy array of the form (||stds||, violation_measure, std_1, std_2, ..., std_n).

        rng (np.random.Generator):
            A numpy random Generator instance for reproducibility.

    Returns:
        np.ndarray:
            A new np.ndarray representing the offspring standard deviations produced through the
            crossover. The offspring have the same dimensionality as the parents.
    """
    assert isinstance(dominant_fitness_metrics, np.ndarray)
    assert isinstance(recessive_fitness_metrics, np.ndarray)
    assert dominant_fitness_metrics.shape == recessive_fitness_metrics.shape

    n_features = dominant_fitness_metrics.shape[0] - 2  # exclude fitness metrics

    # determine the amount of features to inherit from the dominant parent
    softmax = lambda x: np.exp(x) / np.sum(np.exp(x))
    fitness_softmax = softmax(
        [-dominant_fitness_metrics[:2].sum(), -recessive_fitness_metrics[:2].sum()]
    )
    fitness_ratio = np.abs(fitness_softmax[0] - fitness_softmax[1])
    dominant_feature_proportion = max(fitness_ratio, -fitness_ratio + 1)  # opposite proportion
    dominant_feature_amount = max(
        1, int(dominant_feature_proportion * n_features)
    )

    # randomly select the feature indices to inherit from the dominant parent stds
    dominant_features = rng.choice(
        n_features,
        size=dominant_feature_amount,
        replace=False
    )
    # determine the recessive feature indices as the remaining parent stds
    recessive_features = np.setdiff1d(
        np.arange(n_features),
        dominant_features
    )

    # create offspring by combining the parents' features
    offspring_stds = np.zeros(n_features)  # exclude fitness metrics
    offspring_stds[dominant_features] = dominant_fitness_metrics[2:][dominant_features]
    offspring_stds[recessive_features] = recessive_fitness_metrics[2:][recessive_features]

    # mutation stds changes exp(N(0, 0.05))
    std_changes = np.exp(rng.normal(0, 0.05, size=n_features))
    # mutation stds = offspring_stds * std_changes
    offspring_stds *= std_changes
    offspring_stds = np.clip(
        offspring_stds,
        1e-6,
        None
    )

    return offspring_stds

def produce_offspring(
    dominant_fitness_metrics: np.ndarray,
    recessive_fitness_metrics: np.ndarray,
    data: np.ndarray,
    mutation_rate: float,
    rng: np.random.Generator
) -> Individual:
    """
    Produces a new offspring individual by inheriting and mutating the mutation rate
    and mutation standard deviations from the given parent individuals.
    The offsprings data is set as a randomly picked copy of either parent's data,
    with a probability proportional to their fitness values.

    Args:
        dominant_fitness_metrics (np.ndarray):
            A numpy array of the form (||stds||, violation_measure, std_1, std_2, ..., std_n).

        recessive_fitness_metrics (np.ndarray):
            A numpy array of the form (||stds||, violation_measure, std_1, std_2, ..., std_n).

        data (np.ndarray):
            A numpy array of shape (n_samples, n_features) representing the data to be assigned
            to the offspring individual.

        mutation_rate (float):
            A float value representing the mutation rate to be assigned to the offspring individual.

        rng (np.random.Generator):
            A numpy random Generator instance for reproducibility.

    Returns:
        Individual:
            A new individual representing the offspring produced from the inheritance. The standard
            deviations for mutation are inherited and mutated from both parents based on their fitness
            metrics. The mutation rate and data are copied from the dominant parent individual.
    """
    assert isinstance(dominant_fitness_metrics, np.ndarray)
    assert isinstance(recessive_fitness_metrics, np.ndarray)
    assert dominant_fitness_metrics.shape == recessive_fitness_metrics.shape
    assert isinstance(data, np.ndarray)
    assert data.ndim == 2
    assert isinstance(mutation_rate, float)
    assert 0 <= mutation_rate <= 1

    # determine the new mutation stds:
    new_mutation_stds = (dominant_fitness_metrics[2:] + recessive_fitness_metrics[2:]) / 2
    # mutation stds changes exp(N(0, 0.05))
    std_changes = np.exp(rng.normal(0, 0.05, size=new_mutation_stds.shape[0]))
    # mutation stds = new_mutation_stds * std_changes
    new_mutation_stds *= std_changes
    new_mutation_stds = np.clip(
        new_mutation_stds,
        1e-6,
        None
    )

    offspring = Individual(
        data = data.copy(),
        mutation_rate = mutation_rate,
        mutation_stds = new_mutation_stds
    )

    return offspring

def produce_next_generation(
    current_population: list[list[Individual]],
    ranked_metrics: np.ndarray,
    individual_sample_size: int = 15,
    elite_prop: float = 0.05,
    p_combine: float = 0.45,
    X_data: np.ndarray = None,
    X_min: np.ndarray = None,
    X_max: np.ndarray = None,
    X_cat: list = None,
    rng: np.random.Generator = None
) -> list[list[Individual]]:
    """
    Produces the next generation of individuals by selecting elitist individuals that are
    carried over to the next generation without modification, and the remaining individuals
    performing crossover and mutation on the current population based on the provided fitnesses.

    Args:
        current_population (list[list[Individual]]):
            A list of 2D numpy arrays, each representing an individual dataset in the current
            population.

        ranked_metrics (np.ndarray):
            A 2D numpy array of shape (n_population, 2 + n_features) containing the ranked optimization
            metrics for each individual as tuples of (||stds||, violation_measure, std_1, std_2, ..., std_n).

        individual_sample_size (int):
            An integer representing the number of samples per individual.

        elite_prop (float):
            A float value between 0 and 1 representing the proportion of top individuals to
            carry over to the next generation without modification.

        p_combine (float):
            A float representing the probability of combining features from both parents.
            If the random value exceeds this probability, the offspring is created as an exact
            copy of one of the parents, selected randomly weighted by their fitness scores.

        X_min (np.ndarray):
            A 1D numpy array representing the minimum bounds for each feature.

        X_max (np.ndarray):
            A 1D numpy array representing the maximum bounds for each feature.

        X_cat (list):
            A list of nd.arrays representing the possible categories for a categorical feature
            or None if the feature is continuous. The length of the list should be equal to the
            number of features in the individual.

        rng (np.random.Generator):
            A numpy random Generator instance for reproducibility.

    Returns:
        list[list[Individual]]:
            A population of lists of Individual objects, where each list corresponds to an individual
            containing multiple samples. Each Individual has data of shape (n_samples, n_features) and 
            is mutated with the same mutation rate and mutation standard deviation.
    """
    assert isinstance(current_population, list)
    assert all(isinstance(individual_list, list) for individual_list in current_population)
    assert all(isinstance(individual, Individual) for individual_list in current_population for individual in individual_list)
    assert isinstance(ranked_metrics, np.ndarray)
    assert ranked_metrics.ndim == 2
    assert ranked_metrics.shape[0] == len(current_population)
    assert isinstance(individual_sample_size, int)
    assert individual_sample_size > 0
    assert isinstance(elite_prop, float)
    assert 0 < elite_prop < 1
    assert isinstance(p_combine, float)
    assert 0 <= p_combine <= 1
    assert isinstance(X_data, np.ndarray)
    assert X_data.ndim == 2
    assert isinstance(X_min, np.ndarray)
    assert X_min.ndim == 1
    assert X_min.shape[0] == current_population[0][0].data.shape[1]
    assert isinstance(X_max, np.ndarray)
    assert X_max.ndim == 1
    assert X_max.shape[0] == current_population[0][0].data.shape[1]
    assert isinstance(X_cat, list)
    assert len(X_cat) == current_population[0][0].data.shape[1]
    assert all(isinstance(cat, np.ndarray) or cat is None for cat in X_cat)
    assert isinstance(rng, np.random.Generator)

    # read global mutation rate since it is identical for all individuals
    global_mutation_rate = current_population[0][0].mutation_rate

    # determine elitists to carry over to the next generation
    elitists_stds = select_elitists(
        ranked_metrics,
        elite_prop
    )

    # build the next generation starting with the elitists
    next_generation = []
    for stds in elitists_stds:
        sample = []
        for _ in range(individual_sample_size):
            elitist = Individual(
                data = X_data.copy(),
                mutation_rate = global_mutation_rate,
                mutation_stds = stds
            )
            elitist.mutate(
                X_min = X_min,
                X_max = X_max,
                X_cat = X_cat,
                rng = rng
            )
            sample.append(elitist)
        next_generation.append(sample)

    # determine the number of remaining offsprings to produce
    n_offsprings = len(current_population) - len(next_generation)

    # select parent pairs for producing offsprings
    parent_pair_indices = select_parent_indices(
        ranked_metrics,
        n_offsprings,
        rng
    )

    # produce an offspring for each parent pair
    for parent_indices in parent_pair_indices:
        fitness_scores = (
            ranked_metrics[parent_indices[0], :2].sum(),
            ranked_metrics[parent_indices[1], :2].sum()
        )
        # determine dominant and recessive parent
        parent_sorting = np.argsort(fitness_scores)

        # combine parents feature columns with probability p_combine
        if rng.random() < p_combine:
            offspring_stds = crossover_parent_stds(
                dominant_fitness_metrics = ranked_metrics[parent_sorting[0]],
                recessive_fitness_metrics = ranked_metrics[parent_sorting[1]],
                rng = rng
            )

            offspring = Individual(
                data = X_data.copy(),
                mutation_rate = global_mutation_rate,
                mutation_stds = offspring_stds
            )
        else:
            # create offspring by inheriting parents' attributes
            offspring = produce_offspring(
                dominant_fitness_metrics = ranked_metrics[parent_sorting[0]],
                recessive_fitness_metrics = ranked_metrics[parent_sorting[1]],
                data = X_data,
                mutation_rate = global_mutation_rate,
                rng = rng
            )

        sample = []
        for _ in range(individual_sample_size):
            individual_copy = copy.deepcopy(offspring)
            individual_copy.mutate(
                X_min = X_min,
                X_max = X_max,
                X_cat = X_cat,
                rng = rng
            )
            sample.append(individual_copy)

        next_generation.append(sample)

    return next_generation

class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        confidence: float = 0.95
    ):
        """
        Initializes the EarlyStopping object with the specified patience and minimum delta.

        Args:
            patience (int):
                An integer representing the number of consecutive generations to wait for
                improvement before triggering early stopping.

            min_delta (float):
                A float representing the minimum change in fitness value to qualify as an improvement.

            confidence (float):
                A float representing the desired confidence level (between 0 and 1) for the
                drift constraint, such that LCB_Wilson(Drift(X_mutated) >= drift_threshold) >= confidence.
                Default is 0.95.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.confidence = confidence
        self.counter = 0
        self.fitnesses: list[float] = []
        self.best_fitness = np.inf
        self.best_stds = np.empty(0)
        self.mean_probability = -1.0

    def update(
        self,
        fitness: float,
        stds: np.ndarray,
        mean_probability: float
    ):
        """
        Updates the list of best fitnesses with the provided fitness value.

        Args:
            fitness (float):
                A float representing the best fitness value from the current generation.

            stds (np.ndarray):
                A numpy array representing the mutation standard deviations of the best individual.
        
            mean_probability (float):
                A float representing the current generation's mean estimated probability of
                LCB_Wilson(Drift(X_mutated) >= drift_threshold).
        """
        self.fitnesses.append(fitness)

        if self.best_fitness is None or \
            self.best_fitness > fitness + self.min_delta:
            self.best_fitness = fitness
            self.best_stds = stds.copy()
            self.mean_probability = mean_probability

            self.counter = 0
        else:
            self.counter += 1

    def check_early_stopping(self) -> bool:
        """
        Checks whether early stopping should be triggered.

        Returns:
            bool:
                A boolean indicating whether early stopping should be triggered.
        """
        if self.counter >= self.patience and \
            self.mean_probability >= self.confidence:
            return True

        return False

def fitness(
    fitness_metrics: np.ndarray
) -> float:
    """
    Computes the fitness value from the given fitness metrics as the sum of ||stds|| and
    the violation measure, confidence - LCB_Wilson(estimated_probability), evaluated as .
    the maximum of 0 and the violation.
    The computation results from the Lagrangian relaxation of the constrained optimization problem.

    Args:
        fitness_metrics (np.ndarray):
            A 1D numpy array of shape (2,) containing the optimization metrics for an individual
            as a tuple of (||stds||, violation_measure).

    Returns:
        float:
            The computed fitness value.
    """
    assert isinstance(fitness_metrics, np.ndarray)

    return fitness_metrics[0] + max(0, fitness_metrics[1])

def evolve_population(
    initial_population: list[list[Individual]],
    explainer: BaseExplainer,
    reference_explanations: np.ndarray,
    model: BaseModel,
    drift_threshold: float,
    drift_confidence: float,
    n_generations: int = 20,
    elite_prop: float = 0.05,
    p_combine: float = 0.45,
    X_data: np.ndarray = None,
    X_min: np.ndarray = None,
    X_max: np.ndarray = None,
    X_cat: list = None,
    early_stopping_patience: int = 10,
    rng: np.random.Generator = None,
    prediction_threshold: float = None
) -> tuple[np.ndarray, np.ndarray, EarlyStopping, list[tuple[np.ndarray, list[list[Individual]], float]], float]:
    """
    Evolves the given initial population over a specified number of generations
    using genetic algorithm operations such as selection, crossover, and mutation.

    Args:
        initial_population (list[Individual]):
            A list of Individual objects representing the initial population.

        explainer (BaseExplainer):
            An explainer object that can compute feature importance values for instances built with the
            model and initial data to be evaluated.

        reference_explanations (np.ndarray):
            A 2D numpy array of shape (n_samples, n_features) containing the reference
            explanation importance values for comparison.

        model:
            A machine learning model with a `predict()` method that can be used to obtain
            predictions for the individuals' data.

        drift_threshold (float):
            A float value representing the threshold to compare the drift scores against to evaluate
            Probability(drift_score >= threshold).

        drift_confidence (float):
            A float value representing the desired confidence level (between 0 and 1) for the
            drift constraint, such that LCB_Wilson(estimated_probability) >= drift_confidence.

        n_generations (int):
            An integer representing the number of generations to evolve the population.

        elite_prop (float):
            A float representing the proportion of elite individuals to consider for parent
            selection as the major foundation to populate the next generation.

        p_combine (float):
            A float representing the probability of combining features from both parents
            during crossover.

        X_data (np.ndarray):
            A 2D numpy array of shape (n_samples, n_features) representing the data to be used for
            the individuals in the population.

        X_min (np.ndarray):
            A 1D numpy array of shape (n_features,) representing the minimum bounds for each feature.

        X_max (np.ndarray):
            A 1D numpy array of shape (n_features,) representing the maximum bounds for each feature.

        X_cat (list):
            A list of nd.arrays representing the possible categories for a categorical feature
            or None if the feature is continuous. The length of the list should be equal to the
            number of features in the individuals.

        early_stopping_patience (int):
            An integer representing the number of consecutive generations to wait for
            improvement before stopping the evolution process early.

        rng (np.random.Generator):
            A numpy random Generator instance for reproducibility.
        
        prediction_threshold (float):
            A float value representing the threshold for model predictions to be considered valid.

    Returns:
        tuple:
            A tuple containing:
            - A numpy array of the best standard deviations from each generation.
            - A numpy array of the best fitness values from each generation.
            - An EarlyStopping object tracking the evolution process containing the best found
              standard deviations and fitness value.
            - A list of tuples containing the ranked fitness metrics and corresponding population
              for each generation and the mean probability of
              LCB_Wilson(Drift(X_mutated) >= drift_threshold) as logging.
            - A float representing the total time taken for the evolution process in seconds.
    """
    assert isinstance(initial_population, list)
    assert all(isinstance(individual_list, list) for individual_list in initial_population)
    assert all(isinstance(individual, Individual) for individual_list in initial_population for individual in individual_list)
    assert isinstance(explainer, BaseExplainer)
    assert isinstance(reference_explanations, np.ndarray)
    assert reference_explanations.ndim == 2
    assert isinstance(model, BaseModel)
    assert isinstance(drift_threshold, float)
    assert 0 <= drift_threshold
    assert isinstance(drift_confidence, float)
    assert 0 <= drift_confidence <= 1
    assert isinstance(n_generations, int)
    assert n_generations > 0
    assert isinstance(elite_prop, float)
    assert 0 < elite_prop < 1
    assert isinstance(p_combine, float)
    assert 0 <= p_combine <= 1
    assert isinstance(X_data, np.ndarray)
    assert X_data.ndim == 2
    assert isinstance(X_min, np.ndarray)
    assert X_min.ndim == 1
    assert X_min.shape[0] == initial_population[0][0].data.shape[1]
    assert isinstance(X_max, np.ndarray)
    assert X_max.ndim == 1
    assert X_max.shape[0] == initial_population[0][0].data.shape[1]
    assert isinstance(X_cat, list)
    assert len(X_cat) == initial_population[0][0].data.shape[1]
    assert all(isinstance(cat, np.ndarray) or cat is None for cat in X_cat)
    assert isinstance(early_stopping_patience, int)
    assert early_stopping_patience > 0
    assert isinstance(rng, np.random.Generator)
    assert prediction_threshold is not None and isinstance(prediction_threshold, float)

    try:
        time_start = time.time()

        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=1e-4,
            confidence=0.95
        )

        logging = []

        best_stds = []
        best_fitnesses = []
        current_population = copy.deepcopy(initial_population)
        individual_sample_size = len(initial_population[0])

        reference_predictions = model.predict_raw(X_data)

        # calculate the fitness for all individuals in the current population
        fitness_metrics, mean_probability = population_fitness(
            current_population,
            explainer,
            reference_explanations,
            model,
            reference_predictions,
            drift_threshold,
            drift_confidence,
            prediction_threshold
        )

        # rank the individuals' fitness metrics
        ranked_metrics = rank_individual_stds(
            current_population,
            fitness_metrics
        )

        logging.append((ranked_metrics, current_population, mean_probability))

        # store the best individual and fitness from the initial population
        best_stds.append(ranked_metrics[0][2:])
        best_fitnesses.append(fitness(ranked_metrics[0][:2]))

        early_stopping.update(best_fitnesses[-1], best_stds[-1], mean_probability)

        print(f"Initial Population - Best Fitness: {best_fitnesses[-1]:.6f} --- Time: {time.time() - time_start:.2f}s")
        print(f"--- mutation stds: {best_stds[-1]}")

        print("-----")

        # evolve over n_generations
        for _ in range(n_generations):
            generation_start_time = time.time()
            if early_stopping.check_early_stopping():
                print(f"Early stopping triggered before generation {_+1}.")
                break
            else:
                print(f"Starting generation {_+1}/{n_generations}...")

            # produce the next generation
            current_population = produce_next_generation(
                current_population,
                ranked_metrics,
                individual_sample_size,
                elite_prop,
                p_combine,
                X_data,
                X_min,
                X_max,
                X_cat,
                rng
            )

            print(f"--- produced next generation --- Time: {time.time() - generation_start_time:.2f}s")

            # claculate the fitness for all individuals in the current population
            fitness_metrics, mean_probability = population_fitness(
                current_population,
                explainer,
                reference_explanations,
                model,
                reference_predictions,
                drift_threshold,
                drift_confidence,
                prediction_threshold
            )

            # rank the individuals' fitness metrics
            ranked_metrics = rank_individual_stds(
                current_population,
                fitness_metrics
            )

            logging.append((ranked_metrics, current_population, mean_probability))

            print(f"--- calculated fitnesses --- Time: {time.time() - generation_start_time:.2f}s")

            # store the best individual and fitness from the initial population
            best_stds.append(ranked_metrics[0][2:])
            best_fitnesses.append(fitness(ranked_metrics[0][:2]))

            early_stopping.update(best_fitnesses[-1], best_stds[-1], mean_probability)

            print(f"Generation {_+1}/{n_generations} - Best Fitness: {best_fitnesses[-1]:.6f}")
            print(f"--- mutation stds: {best_stds[-1]}")
            print(f"--> completed generation --- Time: {time.time() - generation_start_time:.2f}s (total: {time.time() - time_start:.2f}s)")
    except KeyboardInterrupt:  # allow to stop the evolution process early
        print("Evolution interrupted by user.")

    total_time = time.time() - time_start

    return np.array(best_stds), np.array(best_fitnesses), early_stopping, logging, total_time


class DataPoisoningAttack(BaseAttack):
    """
    A class representing a data poisoning attack.
    
    This attack aims to distort explanations of an explainer by determining the optimal gausian
    white noise in form of the standard deviations for perturbing each feature in the dataset's
    test data using an evolutionary algorithm.

    The population is a list of individual standard deviation distributions that are evaluated
    regarding the potential to produce effictive explanation drifts when applied to the dataset's
    data. To estimate the probability of an explanation drift induced by the white noise of an
    individual in the population, multiple samples are generated for the identical standard deviation
    set of an individual.

    The attack needs to be fitted to the dataset to determine the optimal standard deviations
    before it can be applied to generate poisoned data.
    """
    def __init__(
        self,
        dataset: BaseDataset,
        model: BaseModel,
        explainer: BaseExplainer,
        task: Literal["classification","regression"]="classification",
        random_state: int = None,
        epsilon: float = None
    ):
        """
        Initializes the DataPoisoningAttack object with the specified arguments.

        Args:
            dataset (BaseDataset):
                The dataset to be used for the attack.

            model (BaseModel):
                The model to be attacked.

            explainer (BaseExplainer):
                An explainer object that can compute explanations for data instances.

            task (Literal["classification","regression"]):
                The type of task the model is performing, either "classification" or "regression".

            random_state (int):
                An integer representing the random seed for reproducibility.

            epsilon (float):
                A float value representing the maximum allowed perturbation for each feature.
        """
        super().__init__(model, task, epsilon, stats=[self,"DataPoisoningAttack"],dataset=dataset)
        self.explainer = explainer

        self.rng = np.random.default_rng(seed=random_state)

    def fit(
        self,
        N_GEN: int = 100,
        N_POP: int = 20,
        N_SAMPLE: int = 15,
        INIT_MUTATION_RATE: float = 1.0,
        INIT_STD: float = 0.1,
        P_ELITE: float = 0.05,
        P_COMBINE: float = 0.1,
        DRIFT_THRESHOLD: float = 0.1,
        DRIFT_CONFIDENCE: float = 0.95,
        EARLY_STOPPING_PATIENCE: int = 10,
        EXPLAINER_NUM_SAMPLES: int = 100,
        EVOLUTION_DATA_NUM_SAMPLES: int = 200,
    ):
        """
        Fits the data poisoning attack by using an evolutionary algorithm to find optimal
        mutation gaussian white noise standard deviations for perturbing each feature in
        the dataset's test data.

        Note
        ----
            Be aware that for large datasets and a high number of generations as well as
            samples per individual, the fitting process can take a significant amount of time.
        
        Args:
            N_GEN (int):
                An integer representing the number of generations to evolve the population
                of standard deviations.
        
            N_POP (int):
                An integer representing the size of the population of different standard 
                deviations to evaluate.
        
            N_SAMPLE (int):
                An integer representing the number of samples to generate per individual
                standard deviation set in the population. This is used to estimate the 
                probability of an explanation drift induced by the applied perturbations.
        
            INIT_MUTATION_RATE (float):
                A float value from (0;1] representing the initial mutation rate. This can 
                regulate the proportion of instances in a dataset that are perturbed when
                generating mutated datasets to simulate sparse poisoning attacks if desired.
        
            INIT_STD (float):
                A float value representing the initial standard deviation for the gaussian 
                white noise applied to each feature in the dataset's data. At initialization,
                all individuals in the population start with a slight random variation around
                this value for each feature. 

            P_ELITE (float):
                A float value between 0 and 1 representing the proportion of elite individuals
                to consider for parent selection to directly insert into the next generation
                of individuals.
        
            P_COMBINE (float):
                A float value between 0 and 1 representing the probability of combining standard
                deviation values of two individuals per feature during crossover. If the random
                value exceeds this probability, the offspring is created as an exact copy of
                one of the parents, selected randomly weighted by their fitness scores to populate
                the next generation of individuals.

            DRIFT_THRESHOLD (float):
                A float value to evaluate the induced explanation drift against when estimating
                the probability of an explanation drift for each individual in the population.
                It is used to evaluate whether an individual produces sufficient explanation 
                drift with the constraint:
                    LCB_Wilson(Drift(X_mutated) >= drift_threshold) >= drift_confidence
            
            DRIFT_CONFIDENCE (float):
                A float value between 0 and 1 representing the desired confidence level for the
                evaluated explanation drift constraint:
                    LCB_Wilson(Drift(X_mutated) >= drift_threshold) >= drift_confidence

            EARLY_STOPPING_PATIENCE (int):
                An integer representing the number of consecutive generations to wait for
                improvement before stopping the evolution process early. As improvement factors
                are counted both a better internal fitness value of a standard deviation feature
                distribution and an increased estimated probability of the explanation drift 
                constraint above the desired confidence level.

            EXPLAINER_NUM_SAMPLES (int):
                An integer representing the number of samples to use for the explainer during
                the explanation process.

            EVOLUTION_DATA_NUM_SAMPLES (int):
                An integer representing the number of data samples to use from the dataset's
                test data during the evolution process to evaluate the individuals in the population.
                This can be used to speed up the fitting process on large datasets by using a
                smaller subset of the data for the evolution.
        """
        assert isinstance(N_GEN, int) and N_GEN > 0
        assert isinstance(N_POP, int) and N_POP > 0
        assert isinstance(N_SAMPLE, int) and N_SAMPLE > 0
        assert isinstance(INIT_MUTATION_RATE, float) and 0 < INIT_MUTATION_RATE <= 1
        assert isinstance(INIT_STD, float) and INIT_STD >= 0
        assert isinstance(P_ELITE, float) and 0 < P_ELITE < 1
        assert isinstance(P_COMBINE, float) and 0 <= P_COMBINE <= 1
        assert isinstance(DRIFT_THRESHOLD, float) and DRIFT_THRESHOLD >= 0
        assert isinstance(DRIFT_CONFIDENCE, float) and 0 <= DRIFT_CONFIDENCE <= 1
        assert isinstance(EARLY_STOPPING_PATIENCE, int) and EARLY_STOPPING_PATIENCE > 0
        assert isinstance(EXPLAINER_NUM_SAMPLES, int) and EXPLAINER_NUM_SAMPLES > 0
        self.stats("fit")

        # lower number of samples for explanation during fitting for performance
        original_explainer_num_samples = self.explainer.num_samples
        self.explainer.num_samples = EXPLAINER_NUM_SAMPLES
        
        # determine feature bounds and categorical feature information
        self.X_min, self.X_max = np.array(list(self.dataset.scaled_feature_ranges.values())).T
        cat_mask = self.dataset.categorical_feature_mask
        self.X_cat = list(self.dataset.scaled_categorical_values.values())

        if EVOLUTION_DATA_NUM_SAMPLES < self.dataset.X_test_scaled.shape[0]:
            # sample a subset of the test data for faster evolution
            sampled_indices = self.rng.choice(
                self.dataset.X_test_scaled.shape[0],
                size=EVOLUTION_DATA_NUM_SAMPLES,
                replace=False
            )
            X_data = self.dataset.X_test_scaled.iloc[sampled_indices].values
            y_data = self.dataset.y_test.iloc[sampled_indices].values
        else:
            X_data = self.dataset.X_test_scaled.values
            y_data = self.dataset.y_test.values

        # populate an initial population of std individuals
        initial_population = init_population(
            reference_data=X_data,
            cat_mask=cat_mask,
            population_size=N_POP,
            mutation_rate=INIT_MUTATION_RATE,
            initial_mutation_std=INIT_STD,
            individual_sample_size=N_SAMPLE,
            rng=self.rng
        )

        # compute the reference explanation to compare the drift against
        reference_explanations = self.explainer.explain(X_data)

        # determine prediction threshold relative to the target scaling for regression
        if self.model.task == "regression":
            prediction_threshold = self.epsilon * (y_data.max() - y_data.min())
        else:
            prediction_threshold = self.epsilon

        # start the evolution process
        gen_stds, gen_fitnesses, early_stopping, logging, total_time = evolve_population(
            initial_population=initial_population,
            explainer=self.explainer,
            reference_explanations=reference_explanations,
            model=self.model,
            drift_threshold=DRIFT_THRESHOLD,
            drift_confidence=DRIFT_CONFIDENCE,
            n_generations=N_GEN,
            elite_prop=P_ELITE,
            p_combine=P_COMBINE,
            X_data=X_data,
            X_min=self.X_min,
            X_max=self.X_max,
            X_cat=self.X_cat,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            rng=self.rng,
            prediction_threshold=prediction_threshold
        )

        # save the best found stds for data poisoning on the dataset
        self.scaled_attack_stds = early_stopping.best_stds

        # save the results of the evolution process to json
        results = {
            "generation_stds": gen_stds.tolist(),
            "best_stds": early_stopping.best_stds.tolist(),
            "generation_fitnesses": gen_fitnesses.tolist(),
            "best_fitness": early_stopping.best_fitness,
            "mean_probabilities": [log[2] for log in logging],
            "last_mean_probability": early_stopping.mean_probability,
            "total_time_seconds": total_time,
            "meta": {
                "N_GEN": N_GEN,
                "N_POP": N_POP,
                "N_SAMPLE": N_SAMPLE,
                "DATASET_SHAPE": self.dataset.X_test.shape,
                "INIT_MUTATION_RATE": INIT_MUTATION_RATE,
                "INIT_STD": INIT_STD,
                "P_ELITE": P_ELITE,
                "P_COMBINE": P_COMBINE,
                "DRIFT_THRESHOLD": DRIFT_THRESHOLD,
                "DRIFT_CONFIDENCE": DRIFT_CONFIDENCE,
                "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
                "STOPPED_GENERATION": len(gen_fitnesses),
                "EXPLAINER_SAMPLES_DURING_FIT": self.explainer.num_samples
            }
        }

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        filepath = Path(Path(__file__).parent, '..', '..', '..', 'skripts', 'results', f'DataPoisoningAttack_evolution_results_{timestamp}.json')
        with open(filepath, "w") as f:
            json.dump(results, f, indent=4)

        # restore original explainer number of samples
        self.explainer.num_samples = original_explainer_num_samples

    def generate(
        self,
        x: np.ndarray,
        scaled: bool = True
    ) -> np.ndarray:
        """
        Generates poisoned data by applying gaussian white noise with the learned standard
        deviations to the specified dataset.

        Args:
            x (np.ndarray):
                A 2D numpy array of shape (n_samples, n_features) representing the data to
                be poisoned.

            scaled (bool):
                A boolean indicating whether the provided data is already scaled according
                to the dataset's scaling. If False, the perturbations will be scaled back to
                the original data scale before being applied. Default is True.
        
        Returns:
            np.ndarray:
                A 2D numpy array of shape (n_samples, n_features) representing the poisoned
                data.
        """
        assert hasattr(self, "scaled_attack_stds"), "The attack must be fitted before generating poisoned data."
        assert isinstance(x, np.ndarray) or isinstance(x, pd.DataFrame)
        assert x.ndim == 2
        assert x.shape[1] == self.scaled_attack_stds.shape[0]
        self.stats("generate", x)

        if isinstance(x, pd.DataFrame):
            x = x.values

        x_poisoned = x.copy()

        # generate white noise mutation with feature std
        white_noise = self.rng.normal(
            0,
            self.scaled_attack_stds,
            size=x_poisoned.shape
        )

        if not scaled:
            white_noise *= self.dataset.X_test.values.std(axis=0)

        # create gaussian mutations with respective stds
        for feature in range(x_poisoned.shape[1]):
            # categorical feature mutation
            if self.X_cat[feature] is not None:
                # map each continuous mutation value to the nearest category value
                indices = np.abs(
                    x_poisoned[:, feature, None] + \
                    white_noise[:, feature, None] - \
                    self.X_cat[feature][None, :]
                ).argmin(axis=-1)
                mutations = self.X_cat[feature][indices]

                # apply mutations according to the mask
                x_poisoned[:, feature] = mutations
            # continuous feature mutation
            else:
                # apply mutations according to the mask
                x_poisoned[:, feature] += white_noise[:, feature]

        # clip the mutated data to the feature bounds
        x_poisoned = np.clip(
            x_poisoned,
            self.X_min,
            self.X_max
        )

        # check validity of the poisoned x on every feature
        all_okay, _ = self.is_attack_valid(x, x_poisoned, epsilon=self.epsilon)

        return np.where(all_okay.reshape(-1, 1), x_poisoned, x)

    def _generate(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Generates poisoned data by applying gaussian white noise with the learned standard
        deviations to the specified dataset.

        Args:
            x (np.ndarray):
                A 1D numpy array of shape (n_features,) representing the data to
                be poisoned.
        
        Returns:
            np.ndarray:
                A 1D numpy array of shape (n_features,) representing the poisoned
                data.
        """
        assert hasattr(self, "scaled_attack_stds"), "The attack must be fitted before generating poisoned data."
        assert isinstance(x, np.ndarray) or isinstance(x, pd.DataFrame)
        assert x.ndim == 1
        assert x.shape[0] == self.scaled_attack_stds.shape[0]

        if isinstance(x, pd.DataFrame):
            x = x.values

        # generate poisoned data using the public generate method
        x_poisoned = self.generate(
            x.reshape(1, -1),
            scaled=False
        ).reshape(-1)

        return x_poisoned
