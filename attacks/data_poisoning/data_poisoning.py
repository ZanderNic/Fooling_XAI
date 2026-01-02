import numpy as np
from scipy.stats import kendalltau, norm
import copy
import shap
import matplotlib.pyplot as plt

import time

# Genetic Algorithm Components
def produce_shap_explanations(
    explainer: shap.Explainer,
    dataset: np.ndarray
) -> np.ndarray:
    """
    Produces an array of SHAP explanations for each instance of the given dataset using
    the provided SHAP explainer. The explanations are determined w.r.t. the class 1.

    Parameters
    ----------
    explainer : shap.Explainer
        A SHAP explainer object that can compute SHAP values for instances.

    dataset : np.ndarray
        A numpy array of shape (n_samples, n_features) containing the instances for 
        which local explanations are to be generated.

    Returns
    -------
    np.ndarray
        A 2D numpy array of shape (n_samples, n_features) where each row corresponds to
        the SHAP values for a single instance in the dataset.
    """
    assert isinstance(explainer, shap.Explainer)
    assert isinstance(dataset, np.ndarray)
    assert dataset.ndim == 2

    explanations = explainer(dataset).values[:, :, 1]

    return explanations

def manhattan_distance(
    vec1: np.ndarray,
    vec2: np.ndarray
) -> float:
    """
    Computes the Manhattan distance between two vectors.

    Parameters
    ----------
    vec1 (np.ndarray):
        A 1D numpy array representing the first vector of explanation importance values.

    vec2 (np.ndarray):
        A 1D numpy array representing the second vector of explanation importance values.

    Returns
    -------
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

    Parameters
    ----------
    vec1 (np.ndarray):
        A 3D numpy array of shape (n_population, n_samples, n_features) holding the 
        first vectors of explanation importance values with shape (n_features,).

    vec2 (np.ndarray):
        A 3D numpy array of shape (n_population, n_samples, n_features) holding the
        second vectors of explanation importance values with shape (n_features,).

    Returns
    -------
    float:
        The normalized Kendall Tau distance between the two vectors.
    """
    assert isinstance(vec1, np.ndarray)
    assert isinstance(vec2, np.ndarray)
    assert vec1.ndim >= 2
    assert vec1.shape == vec2.shape

    # produce rankings
    vec1_ranking = np.argsort(abs(vec1), axis=-1, stable=True)
    vec2_ranking = np.argsort(abs(vec2), axis=-1, stable=True)

    # compute kendall tau distance for each instance
    taus = np.array([
        kendalltau(vec1_ranking[pop_i, sample_i], vec2_ranking[pop_i, sample_i])[0] 
        for pop_i in range(vec1.shape[0])
        for sample_i in range(vec1.shape[1])
    ]).reshape(vec1.shape[0], vec1.shape[1])

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

    Parameters
    ----------
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

    Returns
    -------
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

    Parameters
    ----------
    mutation_explanations (np.ndarray):
        A 2D numpy array of shape (n_samples, n_features) containing the explanation
        importance values for the mutated individual.

    reference_explanations (np.ndarray):
        A 2D numpy array of shape (n_samples, n_features) containing the explanation
        importance values for the reference individual.

    Returns
    -------
    np.ndarray:
        A 1D numpy array of shape (n_samples,) containing the computed drift scores.
    """
    loss = manipulation_loss(
        explanation_importance_values = (
            reference_explanations,
            mutation_explanations
        ),
        distance_fn = manhattan_distance,
        order_ranking_fn = kendall_tau_distance,
        weights = (0.5, 0.5)
    )

    return loss

def LCB_Wilson(
    estimated_probability: np.ndarray,
    sample_size: int,
    confidence: float = 0.95
) -> np.ndarray:
    """
    Computes the lower confidence bound using the Wilson score interval.

    Parameters
    ----------
    estimated_probability (np.ndarray):
        A 1D numpy array of shape (n_samples,) containing the estimated probabilities.

    sample_size (int):
        The total number of samples.

    confidence (float):
        The desired confidence level (between 0 and 1).

    Returns
    -------
    np.ndarray:
        A 1D numpy array of shape (n_samples,) containing the computed lower confidence bounds.
    """
    assert isinstance(estimated_probability, np.ndarray)
    assert estimated_probability.ndim == 1
    assert isinstance(sample_size, int)
    assert sample_size > 0
    assert isinstance(confidence, float)
    assert 0 < confidence < 1

    # standard normal confidence interval z_{alpha/2}
    z = norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2 / sample_size
    adjusted_prob = estimated_probability + z**2 / (2 * sample_size)
    lcb = (
        adjusted_prob - \
        (z**2 / (2 * sample_size)) * \
        np.sqrt(
            4 * sample_size * estimated_probability * \
            ( 1 - estimated_probability ) + z**2
        )
    ) / denominator

    return lcb

class Individual:
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
        X_cat: list
    ):
        """
        Mutates the individual data instance by applying N(0, mutation_stds[i]) gaussian white
        noise mutations to each continuous feature and another categorical value to each 
        categorical feature with the individuals given mutation probability.
        The mutated data is clipped to the provided feature bounds.

        Parameters
        ----------
        X_min (np.ndarray):
            A 1D numpy array representing the minimum bounds for each feature.

        X_max (np.ndarray):
            A 1D numpy array representing the maximum bounds for each feature.

        X_cat (list):
            A list of numpy arrays representing the possible categories for a categorical
            feature or None if the feature is continuous. The length of the list should be
            equal to the number of features in the individual.
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

        # create gaussian mutations with respective stds
        for feature in range(self.data.shape[1]):
            # decide which instances to mutate
            mutation_mask = np.random.rand(self.data.shape[0]) < self.mutation_rate

            # generate mutations with feature std
            mutations = np.random.normal(
                0,
                self.mutation_stds[feature],
                size=self.data.shape[0]
            )

            # categorical feature mutation
            if X_cat[feature] is not None:
                # map each continuous mutation value to the nearest category value
                indices = np.abs(mutations[:, None] - X_cat[feature][None, :]).argmin(axis=1)
                mutations = X_cat[feature][indices]

                # apply mutations according to the mask
                self.data[mutation_mask, feature] = mutations[mutation_mask]
            # continuous feature mutation
            else:
                # apply mutations according to the mask
                self.data[mutation_mask, feature] += mutations[mutation_mask]

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
) -> list[list[Individual]]:
    """
    Initilizes a population of n individuals by mutating copies of the provided reference data.

    Parameters
    ----------
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

    Returns
    -------
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
    assert 0 < initial_mutation_std <= 1
    assert isinstance(individual_sample_size, int)
    assert individual_sample_size > 0

    population = []

    # determine random mutation stds for each individual
    mutation_stds = np.clip(np.random.normal(
        loc=initial_mutation_std,
        scale=0.015,
        size=(population_size, reference_data.shape[1])
    ), 0, None)

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
                ]
            )

            individual_list.append(individual)

        population.append(individual_list)

    return population

def estimate_probability(
    drift_scores: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Estimates the probability that the drift scores exceed the given threshold.

    Parameters
    ----------
    drift_scores (np.ndarray):
        A 2D numpy array of shape (n_population, n_samples) containing the globalized drift score
        for each sample of mutation standard deviations in the population. 

    threshold (float):
        A float value representing the threshold to compare the drift scores against to evaluate
        Probability(drift_score >= threshold).

    Returns
    -------
    np.ndarray:
        A 1D numpy array of shape (n_population,) containing the estimated probabilities
        for each individual in the population. 
    """
    assert isinstance(drift_scores, np.ndarray)
    assert drift_scores.ndim == 2
    assert isinstance(threshold, float)
    assert 0 <= threshold

    # Compute the estimated probabilities
    return np.mean(drift_scores >= threshold, axis=1)

def population_fitness(
    population: list[list[Individual]],
    explainer: shap.Explainer,
    reference_explanations: np.ndarray,
    drift_threshold: float = 0.5,
    drift_confidence: float = 0.95
) -> np.ndarray:
    """
    Computes the optimization metrics for a population of individuals. The metrics are based on 
    the individuals' explanation drift scores compared to reference explanations. The values are 
    determined as tuples of (||stds||, violation_measure) where ||stds|| is the 1-norm of the mutation
    standard deviations and violation_measure is the amount by which the lower confidence bound of
    the estimated probability, that the drift scores exceed the given threshold, violates the drift
    confidence constraint defined by drift_confidence.

    Parameters
    ----------
    population (list[list[Individual]]):
        A population of individuals represented as a list of samples of Individual objects where
        each sample share the same underlying standard deviation for mutating the individuals' data.

    explainer (shap.Explainer):
        A SHAP explainer object that can compute SHAP values for instances.

    reference_explanations (np.ndarray):
        A 2D numpy array of shape (n_datapoints, n_features) containing the reference explanation
        importance values for each data feature.

    drift_threshold (float):
        A float value representing the threshold to compare the drift scores against to evaluate
        Probability(drift_score >= threshold).

    drift_confidence (float):
        A float value representing the desired confidence level (between 0 and 1) for the
        drift constraint, such that LCB_Wilson(estimated_probability) >= drift_confidence.

    Returns
    -------
    np.ndarray:
        A 2D numpy array of shape (n_population, 2) with the optimization metrics for each individual
        aggregated over the samples as tuples of (||stds||, violation_measure). 
    """
    assert isinstance(population, list)
    assert all(isinstance(individual_list, list) for individual_list in population)
    assert all(isinstance(individual, Individual) for individual_list in population for individual in individual_list)
    assert all(individual.data.ndim == 2 for individual_list in population for individual in individual_list)
    assert isinstance(explainer, shap.Explainer)
    assert isinstance(reference_explanations, np.ndarray)
    assert reference_explanations.ndim == 2
    assert isinstance(drift_threshold, float)
    assert 0 <= drift_threshold
    assert isinstance(drift_confidence, float)
    assert 0 <= drift_confidence <= 1

    # determine dimensionalities
    n_pop = len(population)
    n_sample_size = len(population[0])
    n_datapoints = population[0][0].data.shape[0]
    n_features = population[0][0].data.shape[1]

    print("Start population fitness calculation:")
    start_time = time.time()  # LOGGING

    # unify all individuals' data in a single array without adding a new axis
    pop = np.concatenate(
        [individual.data for individual_list in population for individual in individual_list]
    )

    print(f"\tConcatenating population took {time.time() - start_time} seconds.")
    section_time = time.time()  # LOGGING

    # produce SHAP explanations for the entire population
    explanations = produce_shap_explanations(
        explainer,
        pop
    ).reshape(n_pop, n_sample_size, n_datapoints, n_features)

    # average explanations over data points for each individual and sample
    mean_explanations = np.mean(
        explanations,
        axis=-2
    )

    print(f"\tProducing SHAP explanations took {time.time() - section_time} seconds.")
    section_time = time.time()  # LOGGING

    # average reference explanations over data points
    mean_reference_explanations = np.mean(
        reference_explanations,
        axis=0
    )

    # repeat the reference explanations for all individuals
    # to match the above explanations shape
    reference_explanations_rep = np.tile(
        mean_reference_explanations,
        (n_pop, n_sample_size, 1)
    )

    print(f"\tProducing SHAP reference explanations took {time.time() - section_time} seconds.")
    section_time = time.time()  # LOGGING

    # calculate the drift scores for all samples
    drift_scores = explanation_drift(
        mutation_explanations = mean_explanations,
        reference_explanations = reference_explanations_rep
    )

    print(f"\tCalculating drift scores took {time.time() - section_time} seconds.")
    section_time = time.time()  # LOGGING

    # estimate the probabilities that the drift scores exceed the given threshold
    estimated_probabilities = estimate_probability(
        drift_scores = drift_scores,
        threshold = drift_threshold
    )

    print(f"\tCalculating estimated probs took {time.time() - section_time} seconds.")
    section_time = time.time()  # LOGGING

    # determine the lower confidence interval bound for the estimated probabilities
    lcb = LCB_Wilson(
        estimated_probability = estimated_probabilities,
        sample_size = n_sample_size
    )

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

    return fitnesses

def rank_individual_stds(
    population: list[list[Individual]],
    fitnesses: np.ndarray
) -> np.ndarray:
    """
    Ranks the individuals' standard deviations according to their fitnesses. The ranking is done
    primarily by the violation componnt of the fitnesses ascendingly and secondarily by the
    1-norm of the individuals' standard deviations ascendingly.

    Parameters
    ----------
    population (list[list[Individual]]):
        A population of individuals represented as a list of samples of Individual objects where
        each sample share the same underlying standard deviation for mutating the individuals' data.

    fitnesses (np.ndarray):
        A 2D numpy array of shape (n_population, 2) containing the optimization metrics for each
        individual as tuples of (||stds||, violation_measure).

    Returns
    -------
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

    # sort fitnesses primarily by the second column (violation) ascending,
    # secondarily by the first column (||stds||) ascending
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
) -> list[Individual]:
    """
    Selects the top `elite_prop` proportion of individuals from the population based on their
    fitness scores.

    Parameters
    ----------
    ranked_metrics (np.ndarray):
        A 2D numpy array of shape (n_population, 2 + n_features) containing the ranked optimization
        metrics for each individual as tuples of (||stds||, violation_measure, std_1, std_2, ..., std_n).

    elite_prop (float):
        A float value between 0 and 1 representing the proportion of individuals to select as elitists.

    Returns
    -------
    list[Individual]:
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
    n_pairs: int
) -> np.ndarray:
    """
    Selects parent indices for crossover from the individuals in the population
    based on their fitness values.
    The probability of selecting an individual is proportional to its fitness value
    using a softmax distribution.

    Parameters
    ----------
    ranked_metrics (np.ndarray):
        A 2D numpy array of shape (n_population, 2 + n_features) containing the ranked optimization
        metrics for each individual as tuples of (||stds||, violation_measure, std_1, std_2, ..., std_n).

    n_pairs (int):
        The number of parent pairs to select.

    Returns
    -------
    np.ndarray:
        A 2D numpy array of shape (n_population, 2) containing pairs of selected parent indices
        for crossover.
    """
    assert isinstance(ranked_metrics, np.ndarray)
    assert ranked_metrics.ndim == 2
    assert isinstance(n_pairs, int)
    assert n_pairs > 0

    # softmax weights for all individuals
    softmax = lambda x: np.exp(x) / np.sum(np.exp(x))
    individual_weights = softmax(-1 * (ranked_metrics[:, 0] + ranked_metrics[:, 1]))

    # determine parent pairs from dominant and recessive individuals
    parent_pair_indices = np.random.choice(
        np.arange(ranked_metrics.shape[0]),
        size=(n_pairs, 2),
        replace=True,
        p=individual_weights
    )

    return parent_pair_indices

def crossover_parent_stds(
    dominant_fitness_metrics: np.ndarray,
    recessive_fitness_metrics: np.ndarray
) -> np.ndarray:
    """
    Performs crossover of standard deviations between the given parent individuals to produce
    offspring standard deviations with values inherited from both parents based on their
    fitness metrics.

    Parameters
    ----------
    dominant_fitness_metrics (np.ndarray):
        A numpy array of the form (||stds||, violation_measure, std_1, std_2, ..., std_n).

    recessive_fitness_metrics (np.ndarray):
        A numpy array of the form (||stds||, violation_measure, std_1, std_2, ..., std_n).

    n_features (int):
        The number of features in the individuals' data.

    Returns
    -------
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

    # randomly select the features to inherit from the dominant parent stds
    dominant_features = np.random.choice(
        n_features,
        size=dominant_feature_amount,
        replace=False
    )
    # determine the recessive features as the remaining features
    recessive_features = np.setdiff1d(
        np.arange(n_features),
        dominant_features
    )

    # create offspring by combining the parents' features
    offspring_stds = np.zeros_like(dominant_fitness_metrics[2:])  # exclude fitness metrics
    offspring_stds[dominant_features] = dominant_fitness_metrics[2:][dominant_features]
    offspring_stds[recessive_features] = recessive_fitness_metrics[2:][recessive_features]

    # mutation stds changes exp(N(0, 0.1))
    std_changes = np.exp(np.random.normal(0, 0.05, size=offspring_stds.shape[0]))
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
    mutation_rate: float
) -> Individual:
    """
    Produces a new offspring individual by inheriting and mutating the mutation rate
    and mutation standard deviations from the given parent individuals.
    The offsprings data is set as a randomly picked copy of either parent's data,
    with a probability proportional to their fitness values.

    Parameters
    ----------
    dominant_fitness_metrics (np.ndarray):
        A numpy array of the form (||stds||, violation_measure, std_1, std_2, ..., std_n).

    recessive_fitness_metrics (np.ndarray):
        A numpy array of the form (||stds||, violation_measure, std_1, std_2, ..., std_n).

    data (np.ndarray):
        A numpy array of shape (n_samples, n_features) representing the data to be assigned
        to the offspring individual.

    mutation_rate (float):
        A float value representing the mutation rate to be assigned to the offspring individual.

    Returns
    -------
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
    # mutation stds changes exp(N(0, 0.1))
    std_changes = np.exp(np.random.normal(0, 0.1, size=new_mutation_stds.shape[0]))
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
    X_cat: list = None
) -> list[list[Individual]]:
    """
    Produces the next generation of individuals by selecting elitist individuals that are
    carried over to the next generation without modification, and the remaining individuals
    performing crossover and mutation on the current population based on the provided fitnesses.

    Parameters
    ----------
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

    Returns
    -------
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
    assert isinstance(X_min, np.ndarray)
    assert X_min.ndim == 1
    assert X_min.shape[0] == current_population[0][0].data.shape[1]
    assert isinstance(X_max, np.ndarray)
    assert X_max.ndim == 1
    assert X_max.shape[0] == current_population[0][0].data.shape[1]
    assert isinstance(X_cat, list)
    assert len(X_cat) == current_population[0][0].data.shape[1]
    assert all(isinstance(cat, np.ndarray) or cat is None for cat in X_cat)

    # read global mutation rate
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
                X_cat = X_cat
            )
            sample.append(elitist)
        next_generation.append(sample)

    # determine the number of remaining offsprings to produce
    n_offsprings = len(current_population) - len(next_generation)

    # select parent pairs for producing offsprings
    parent_pair_indices = select_parent_indices(
        ranked_metrics,
        n_offsprings
    )

    # produce an offspring for each parent pair
    for parent_indices in parent_pair_indices:
        fitness_scores = (
            ranked_metrics[parent_indices[0], 0] + ranked_metrics[parent_indices[0], 1],
            ranked_metrics[parent_indices[1], 0] + ranked_metrics[parent_indices[1], 1]
        )
        # determine dominant and recessive parent
        parent_sorting = np.argsort(fitness_scores)

        # combine parents feature columns with probability p_combine
        if np.random.rand() < p_combine:
            offspring_stds = crossover_parent_stds(
                dominant_fitness_metrics = ranked_metrics[parent_sorting[1]],
                recessive_fitness_metrics = ranked_metrics[parent_sorting[0]],
            )

            offspring = Individual(
                data = X_data.copy(),
                mutation_rate = global_mutation_rate,
                mutation_stds = offspring_stds
            )
        else:
            # create offspring by inheriting parents' attributes
            offspring = produce_offspring(
                dominant_fitness_metrics = ranked_metrics[parent_sorting[1]],
                recessive_fitness_metrics = ranked_metrics[parent_sorting[0]],
                data = X_data,
                mutation_rate = global_mutation_rate            
            )

        sample = []
        for _ in range(individual_sample_size):
            individual_copy = copy.deepcopy(offspring)
            individual_copy.mutate(
                X_min = X_min,
                X_max = X_max,
                X_cat = X_cat
            )
            sample.append(individual_copy)

        next_generation.append(sample)

    return next_generation

class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4
    ):
        """
        Initializes the EarlyStopping object with the specified patience and minimum delta.

        Parameters
        ----------
        patience (int):
            An integer representing the number of consecutive generations to wait for
            improvement before triggering early stopping.

        min_delta (float):
            A float representing the minimum change in fitness value to qualify as an improvement.
        """
        self.patience = patience
        self.counter = 0
        self.min_delta = min_delta
        self.fitnesses: list[float] = []
        self.best_fitness = None
        self.best_stds = None

    def update(
        self,
        fitness: float,
        stds: np.ndarray
    ):
        """
        Updates the list of best fitnesses with the provided fitness value.

        Parameters
        ----------
        fitness (float):
            A float representing the best fitness value from the current generation.

        stds (np.ndarray):
            A numpy array representing the mutation standard deviations of the best individual.
        """
        self.fitnesses.append(fitness)

        if self.best_fitness is None or \
            self.best_fitness > fitness + self.min_delta:
            self.best_fitness = fitness
            self.best_stds = stds.copy()

            self.counter = 0
        else:
            self.counter += 1

    def check_early_stopping(self) -> bool:
        """
        Checks whether early stopping should be triggered.

        Returns
        -------
        bool:
            A boolean indicating whether early stopping should be triggered.
        """
        if self.counter >= self.patience:
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

    Parameters
    ----------
    fitness_metrics (np.ndarray):
        A 1D numpy array of shape (2,) containing the optimization metrics for an individual
        as a tuple of (||stds||, violation_measure).

    Returns
    -------
    float:
        The computed fitness value.
    """
    assert isinstance(fitness_metrics, np.ndarray)

    return fitness_metrics[0] + max(0, fitness_metrics[1])

def evolve_population(
    initial_population: list[Individual],
    explainer: shap.Explainer,
    reference_explanations: np.ndarray,
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
) -> tuple[list[np.ndarray], list[float], EarlyStopping, list[tuple[np.ndarray, list[list[Individual]]]]]:
    """
    Evolves the given initial population over a specified number of generations
    using genetic algorithm operations such as selection, crossover, and mutation.

    Parameters
    ----------
    initial_population (list[Individual]):
        A list of Individual objects representing the initial population.

    explainer (shap.Explainer):
        A SHAP explainer object that can compute SHAP values for instances built with the
        model and initial data to be evaluated.

    reference_explanations (np.ndarray):
        A 2D numpy array of shape (n_samples, n_features) containing the reference
        explanation importance values for comparison.

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

    X_min (np.ndarray):
        A 1D numpy array representing the minimum bounds for each feature.

    X_max (np.ndarray):
        A 1D numpy array representing the maximum bounds for each feature.

    X_cat (list):
        A list of nd.arrays representing the possible categories for a categorical feature
        or None if the feature is continuous. The length of the list should be equal to the
        number of features in the individuals.

    early_stopping_patience (int):
        An integer representing the number of consecutive generations to wait for
        improvement before stopping the evolution process early.

    Returns
    -------
    tuple:
        A tuple containing:
        - A list of the best standard deviations from each generation.
        - A list of the best fitness values from each generation.
        - An EarlyStopping object tracking the evolution process containing the best found
          standard deviations and fitness value.
        - A list of tuples containing the ranked fitness metrics and corresponding population
          for each generation as logging.
    """
    assert isinstance(initial_population, list)
    assert all(isinstance(individual_list, list) for individual_list in initial_population)
    assert all(isinstance(individual, Individual) for individual_list in initial_population for individual in individual_list)
    assert isinstance(explainer, shap.Explainer)
    assert isinstance(reference_explanations, np.ndarray)
    assert reference_explanations.ndim == 2
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

    try:
        time_start = time.time()

        early_stopping = EarlyStopping(patience=10, min_delta=1e-4)

        logging = []

        best_stds = []
        best_fitnesses = []
        current_population = copy.deepcopy(initial_population)
        individual_sample_size = len(initial_population[0])

        # claculate the fitness for all individuals in the current population
        fitness_metrics = population_fitness(
            current_population,
            explainer,
            reference_explanations,
            drift_threshold,
            drift_confidence
        )

        # rank the individuals' fitness metrics
        ranked_metrics = rank_individual_stds(
            current_population,
            fitness_metrics
        )

        logging.append((ranked_metrics, current_population))

        print(f"Initial Population - Best Fitness: {fitness(ranked_metrics[0][:2]):.6f} --- Time: {time.time() - time_start:.2f}s")

        # store the best individual and fitness from the initial population
        best_stds.append(ranked_metrics[0][2:])
        best_fitnesses.append(fitness(ranked_metrics[0][:2]))

        early_stopping.update(best_fitnesses[-1], best_stds[-1])

        print(f"before evolution --- Time: {time.time() - time_start:.2f}s")

        # evolve over n_generations
        for _ in range(n_generations):
            if early_stopping.check_early_stopping():
                print(f"Early stopping triggered before generation {_+1}.")
                break

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
                X_cat
            )

            print(f"--- produced next generation --- Time: {time.time() - time_start:.2f}s")

            # claculate the fitness for all individuals in the current population
            fitness_metrics = population_fitness(
                current_population,
                explainer,
                reference_explanations,
                drift_threshold,
                drift_confidence
            )

            # rank the individuals' fitness metrics
            ranked_metrics = rank_individual_stds(
                current_population,
                fitness_metrics
            )

            logging.append((ranked_metrics, current_population))

            print(f"--- calculated fitnesses --- Time: {time.time() - time_start:.2f}s")

            # store the best individual and fitness from the initial population
            best_stds.append(ranked_metrics[0][2:])
            best_fitnesses.append(fitness(ranked_metrics[0][:2]))

            early_stopping.update(best_fitnesses[-1], best_stds[-1])

            print(f"Generation {_+1}/{n_generations} - Best Fitness: {best_fitnesses[-1]:.6f}")
            print(f"--- mutation stds: {best_stds[-1]}")
            print(f"--> completed generation --- Time: {time.time() - time_start:.2f}s")
    except KeyboardInterrupt:
        print("Evolution interrupted by user.")

    return best_stds, best_fitnesses, early_stopping, logging


# Plotting
def plot_evaluation(
    global_reference_explanations: np.ndarray,
    global_mutation_explanations: np.ndarray,
    standard_deviations: np.ndarray,
    feature_names: list[str],
    drift_threshold: float,
    drift_confidence: float
) -> plt.Figure:
    """
    Plots a bar chart comparing the average SHAP explanations of the reference data
    and the mutated data, and prints the standard deviations used for mutation.

    Parameters
    ----------
    global_reference_explanations (np.ndarray):
        A 1D numpy array containing the average SHAP explanations for the reference data.

    global_mutation_explanations (np.ndarray):
        A 1D numpy array containing the average SHAP explanations for the mutated data.

    standard_deviations (np.ndarray):
        A 1D numpy array containing the standard deviations used for mutatating each feature.

    feature_names (list[str]):
        A list of strings representing the names of the features in the dataset.

    drift_threshold (float):
        A float value representing the drift threshold used in the evolution optimization.

    drift_confidence (float):
        A float value representing the drift confidence used in the evolution optimization.

    Returns
    -------
    plt.Figure:
        A matplotlib Figure object containing the generated bar chart.
    """
    x = np.arange(len(feature_names))
    bar_width = 0.3

    fig = plt.figure(figsize=(12, 6), dpi=400)

    # plot bar comparisons of SHAP values
    plt.bar(
        x=x - bar_width/2,
        height=global_reference_explanations,
        width=bar_width,
        color='blue',
        alpha=0.8,
        label='Reference'
    )

    plt.bar(
        x=x + bar_width/2,
        height=global_mutation_explanations,
        width=bar_width,
        color='red',
        alpha=0.8,
        label='Mutation'
    )

    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')

    # show mutation std under each x-tick, rotated vertically
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.15
    # extend lower ylim to make room for the text
    ax.set_ylim(ymin - pad, ymax)

    for xi, s in enumerate(standard_deviations):
        ax.text(
            x[xi],
            ymin - pad * 0.5,
            f"{s:.4f}",
            ha='center',
            va='center',
            rotation=90,
            fontsize=9,
            color='black'
        )

    # replace default tick labels with alternating-height labels to avoid overlap
    ax.set_xticks(x)
    ax.set_xticklabels([])

    # positions in axis coordinates (0..1) for alternating label heights below the axis
    y_top = -0.02
    y_bottom = -0.06

    for i, name in enumerate(feature_names):
        y = y_top if i % 2 == 0 else y_bottom
        ax.text(
            x[i],
            y,
            name,
            ha='center',
            va='top',
            rotation=0,
            transform=ax.get_xaxis_transform(),
            fontsize=9
        )

    plt.title(
        fr"Average SHAP Explanations for "
        fr"$\min_{{\boldsymbol{{\sigma}}}} \|\boldsymbol{{\sigma}}\| \text{{ s.t. }} "
        fr"\mathbb{{P}}(\mathcal{{Drift}}(\tilde{{X}}_{{\boldsymbol{{\sigma}}}}) "
        fr"\geq {drift_threshold:.2f}) \geq {drift_confidence:.2f}$",
        fontsize=14
    )
    plt.ylabel("SHAP Value")
    plt.xlabel(
        "Features\n(Applied mutation standard deviations $\\sigma_f$ shown above names)",
    )
    ax.xaxis.set_label_coords(0.5, -0.12)

    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    return fig
