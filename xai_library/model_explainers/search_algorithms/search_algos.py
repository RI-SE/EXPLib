import random
from itertools import product
from math import log, floor
from bayes_opt import BayesianOptimization  ##pip install bayesian-optimization
import numpy as np
import matplotlib.pyplot as plt


def random_search(parameter_eval, param_space, num_iterations: int):
    """
    Perform random search over parameter space.
    """
    results = []
    for _ in range(num_iterations):
        params = {param: random.choice(values) for param, values in param_space.items()}
        result = parameter_eval(params)
        results.append((params, result))
    return results


def grid_search(parameter_eval, param_space):
    """
    Perform grid search over parameter space.
    """
    keys = list(param_space.keys())
    values = list(param_space.values())
    param_combinations = list(product(*values))

    results = []
    for params in param_combinations:
        param_dict = {keys[i]: params[i] for i in range(len(keys))}
        result = parameter_eval(param_dict)
        results.append((param_dict, result))
    return results


def successive_halving(parameter_eval, param_space, eta: int = 3, R: int = 81):
    """
    Simplified Successive Halving algorithm.
    param_space: list of parameter configurations (dicts)
    eta: reduction factor
    R: total budget (number of evaluations)
    """
    if not isinstance(param_space, list):
        raise ValueError("param_space must be a list of parameter dicts for successive halving.")

    results = []
    s_max = floor(log(R, eta))
    B = (s_max + 1) * R  # total budget approximation

    for s in reversed(range(s_max + 1)):
        n = int(B / R * eta**s / (s + 1))
        configs = random.sample(param_space, min(n, len(param_space)))

        scores = []
        for config in configs:
            score = parameter_eval(config)
            scores.append((config, score))
        # Keep top-performing configs for next stage
        configs = [c for c, _ in sorted(scores, key=lambda x: x[1], reverse=True)]
        results.extend(scores)

        # Narrow down for next round
        param_space = configs[:max(1, len(configs)//eta)]

    return results


def bayesian_optimization(parameter_eval, param_space, num_iterations: int):
    """
    Run Bayesian Optimization using bayes_opt library.
    param_space: dict of {'param': (low, high)} bounds
    """
    def objective_function(**params):
        return parameter_eval(params)

    bo = BayesianOptimization(
        f=objective_function,
        pbounds=param_space,
        verbose=0,
        random_state=42
    )

    init_points = min(5, num_iterations // 2)
    bo.maximize(init_points=init_points, n_iter=num_iterations - init_points)

    results = [(res['params'], res['target']) for res in bo.res]
    return results

def genetic_search(parameter_eval, param_space, population_size=20, generations=10, mutation_rate=0.1):
    keys = list(param_space.keys())
    population = [
        {k: random.choice(param_space[k]) for k in keys}
        for _ in range(population_size)
    ]

    def crossover(parent1, parent2):
        return {
            k: random.choice([parent1[k], parent2[k]]) for k in keys
        }

    def mutate(individual):
        for k in keys:
            if random.random() < mutation_rate:
                individual[k] = random.choice(param_space[k])
        return individual

    results = []
    for g in range(generations):
        scored = [(ind, parameter_eval(ind)) for ind in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        results.extend(scored)
        top_k = scored[:population_size // 2]
        parents = [ind for ind, _ in top_k]
        children = []
        for _ in range(population_size - len(parents)):
            p1, p2 = random.sample(parents, 2)
            child = mutate(crossover(p1, p2))
            children.append(child)
        population = parents + children

    return results


def swarm_search(parameter_eval, param_bounds, num_particles=15, num_iterations=20, w=0.7, c1=1.4, c2=1.4):
    """
    Simple continuous PSO implementation.
    param_bounds: dict of parameter: (min, max)
    """
    keys = list(param_bounds.keys())
    dim = len(keys)

    # Initialize positions and velocities
    X = np.array([[random.uniform(*param_bounds[k]) for k in keys] for _ in range(num_particles)])
    V = np.zeros_like(X)
    personal_best = X.copy()
    personal_best_scores = np.array([parameter_eval(dict(zip(keys, x))) for x in X])

    global_best = personal_best[np.argmax(personal_best_scores)]
    global_best_score = np.max(personal_best_scores)

    results = []

    for it in range(num_iterations):
        for i in range(num_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            V[i] = w * V[i] + c1 * r1 * (personal_best[i] - X[i]) + c2 * r2 * (global_best - X[i])
            X[i] += V[i]

            # Clamp positions to bounds
            for j, k in enumerate(keys):
                X[i, j] = np.clip(X[i, j], *param_bounds[k])

            score = parameter_eval(dict(zip(keys, X[i])))
            if score > personal_best_scores[i]:
                personal_best[i] = X[i].copy()
                personal_best_scores[i] = score

            if score > global_best_score:
                global_best = X[i].copy()
                global_best_score = score

            results.append((dict(zip(keys, X[i])), score))

    return results


def simulated_annealing(parameter_eval, param_space, initial_temp=1.0, cooling_rate=0.95, iterations=100):
    keys = list(param_space.keys())
    current = {k: random.choice(param_space[k]) for k in keys}
    current_score = parameter_eval(current)
    best = current.copy()
    best_score = current_score
    results = [(current, current_score)]

    temp = initial_temp
    for _ in range(iterations):
        neighbor = current.copy()
        k = random.choice(keys)
        neighbor[k] = random.choice(param_space[k])
        neighbor_score = parameter_eval(neighbor)

        delta = neighbor_score - current_score
        if delta > 0 or random.random() < np.exp(delta / (temp + 1e-9)):
            current, current_score = neighbor, neighbor_score
            if neighbor_score > best_score:
                best, best_score = neighbor.copy(), neighbor_score

        results.append((current.copy(), current_score))
        temp *= cooling_rate

    return results

def plot_convergence(results_dict):
    plt.figure(figsize=(7, 5))
    for label, res in results_dict.items():
        scores = [r[1] for r in res]
        best_scores = np.maximum.accumulate(scores)
        plt.plot(best_scores, label=label)
    plt.title("Convergence of Search Algorithms")
    plt.xlabel("Iteration")
    plt.ylabel("Best Score So Far")
    plt.legend()
    plt.tight_layout()
    plt.show()