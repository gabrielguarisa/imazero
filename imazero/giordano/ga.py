from math import sqrt, ceil
import numpy as np
from random import randint, choices, random
from imazero.utils import argsort
from imazero.wisard import WiSARD
from sklearn.model_selection import train_test_split


class GiordanoGeneticAlgorithm:
    def __init__(
        self,
        tuple_size,
        entry_size,
        population_size,
        theta_r,
        theta_u,
        num_exec=5,
        max_ittr=100,
        lag=5,
        validation_size=0.3,
    ):
        self.tuple_size = tuple_size
        self.entry_size = entry_size
        self.population_size = population_size
        self.theta_r = theta_r
        self.theta_u = theta_u
        self.num_exec = num_exec
        self.max_ittr = max_ittr
        self.lag = lag
        self.validation_size = validation_size

    def _generate_initial_population(self):
        return [
            np.arange(self.entry_size).reshape((-1, self.tuple_size))
            for _ in range(self.population_size)
        ]

    def mutation(self, mapping):
        new_mapping = mapping.copy()
        last_tuple = len(mapping) - 1
        for i in range(self.tuple_size):
            pos_a = randint(0, last_tuple)
            pos_b = randint(0, last_tuple)
            new_mapping[pos_a][i], new_mapping[pos_b][i] = (
                new_mapping[pos_b][i],
                new_mapping[pos_a][i],
            )

        return new_mapping

    def crossover(self, mapping_a, mapping_b):
        half = ceil(len(mapping_a) / 2.0)
        return (
            np.array([*mapping_a[:half], *mapping_b[half:]]),
            np.array([*mapping_b[:half], *mapping_a[half:]]),
        )

    def fitness_func(self, X, y, mappings):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_size, stratify=y
        )

        return [
            WiSARD(mapping).fit(X_train, y_train).score(X_val, y_val)
            for mapping in mappings
        ]

    def selection(self, population, k=2):
        return choices(population, k=k)

    def run(self, X, y):
        t = 0
        population = self._generate_initial_population()
        fitness = self.fitness_func(X, y, population)
        past_mean = np.mean(fitness)
        diff_counter = 0

        while t < self.max_ittr and diff_counter < self.lag:
            t += 1
            offspring_population = []

            for _ in range(self.num_exec):
                choice = random()

                parent_a, parent_b = self.selection(population)
                if choice < self.theta_r:
                    child, _ = self.crossover(parent_a, parent_b)
                elif choice < self.theta_r + self.theta_u:
                    child = self.mutation(parent_a)
                else:
                    child = parent_a.copy()

                offspring_population.append(child)

            offspring_fitness = self.fitness_func(X, y, offspring_population)
            temp_fitness = [*fitness, *offspring_fitness]
            temp_population = [*population, *offspring_population]
            best_elements = argsort(temp_fitness)[: self.population_size]

            population = [temp_population[i] for i in best_elements]
            fitness = [temp_fitness[i] for i in best_elements]
            fitness_mean = np.mean(fitness)

            if fitness_mean == past_mean:
                diff_counter += 1
            else:
                diff_counter = 0
                past_mean = fitness_mean

            print(
                "t:",
                t,
                "|| fitness:",
                past_mean,
                "|| std:",
                np.std(fitness),
                "|| dc:",
                diff_counter,
            )

        return population, t
