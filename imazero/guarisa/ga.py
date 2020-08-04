from math import sqrt, ceil
import numpy as np
from random import randint, choices, random
from imazero.utils import get_bin, get_int, change_char, random_mapping, argsort
from imazero.wisard import WiSARD
from sklearn.model_selection import train_test_split


class GuarisaGeneticAlgorithm:
    def __init__(
        self,
        tuple_size,
        entry_size,
        population_size,
        theta=0.8,
        num_exec=5,
        max_ittr=100,
        lag=5,
        validation_size=0.3,
    ):
        self.tuple_size = tuple_size
        self.entry_size = entry_size
        self.population_size = population_size
        self.num_exec = num_exec
        self.max_ittr = max_ittr
        self.lag = lag
        self.validation_size = validation_size
        self.theta = theta

    def _generate_random_population(self):
        return [
            random_mapping(self.tuple_size, self.entry_size)
            for _ in range(self.population_size)
        ]

    def _mapping_to_bin(self, mapping):
        num_bits = ceil(sqrt(self.entry_size))
        bin_mapping = []
        for i in range(len(mapping)):
            bin_mapping.append([])
            for j in range(len(mapping[i])):
                bin_mapping[i].append(get_bin(mapping[i][j], num_bits))

        return bin_mapping

    def _bin_to_mapping(self, bin_mapping):
        mapping = []
        for i in range(len(bin_mapping)):
            mapping.append([])
            for j in range(len(bin_mapping[i])):
                value = get_int(bin_mapping[i][j])
                mapping[i].append(
                    value
                    if value < self.entry_size
                    else randint(0, self.entry_size - 1)
                )

        return np.array(mapping, dtype=np.uint32)

    def mutation(self, mapping):
        threshold = 1.0 / len(mapping)
        rand_values = np.random.rand(len(mapping))

        for i in range(len(mapping)):
            if rand_values[i] < threshold:
                mapping[i] = np.random.randint(self.entry_size, size=self.tuple_size)

        return mapping

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

    def selection(self, population, fitness, k=2):
        weigths = np.array(fitness) / np.sum(fitness)
        return choices(population, weights=weigths, k=k)

    def run(self, X, y):
        t = 0
        population = self._generate_random_population()
        fitness = self.fitness_func(X, y, population)
        past_mean = np.mean(fitness)
        diff_counter = 0

        while t < self.max_ittr and diff_counter < self.lag:
            t += 1
            offspring_population = []

            for _ in range(self.num_exec):
                choice = random()

                parent_a, parent_b = self.selection(population, fitness)
                if choice < self.theta:
                    child_a, child_b = self.crossover(parent_a, parent_b)
                    offspring_population.append(child_a)
                    offspring_population.append(child_b)
                else:
                    child = self.mutation(parent_a)
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


class NewGeneticAlgorithm:
    def __init__(
        self,
        tuple_size,
        entry_size,
        population_size,
        recognized_weight,
        recognized_rejected_weight,
        rejected_weight,
        misclassified_weight,
        theta=0.8,
        num_exec=5,
        max_ittr=100,
        lag=5,
        validation_size=0.3,
    ):
        self.tuple_size = tuple_size
        self.entry_size = entry_size
        self.population_size = population_size
        self.num_exec = num_exec
        self.max_ittr = max_ittr
        self.lag = lag
        self.validation_size = validation_size
        self.theta = theta
        self.recognized_weight = recognized_weight
        self.recognized_rejected_weight = recognized_rejected_weight
        self.rejected_weight = rejected_weight
        self.misclassified_weight = misclassified_weight

    def _generate_random_population(self):
        return [
            random_mapping(self.tuple_size, self.entry_size)
            for _ in range(self.population_size)
        ]

    def mutation(self, mapping):
        threshold = 1.0 / len(mapping)
        rand_values = np.random.rand(len(mapping))

        for i in range(len(mapping)):
            if rand_values[i] < threshold:
                mapping[i] = np.random.randint(self.entry_size, size=self.tuple_size)

        return mapping

    def crossover(self, mapping_a, mapping_b):
        half = ceil(len(mapping_a) / 2.0)
        return (
            np.array([*mapping_a[:half], *mapping_b[half:]]),
            np.array([*mapping_b[:half], *mapping_a[half:]]),
        )

    def o_func(self, recognized, recognized_rejected, rejected, misclassified):
        return (
            recognized * self.recognized_weight
            + recognized_rejected * self.recognized_rejected_weight
            + misclassified * self.misclassified_weight
            + rejected * self.rejected_weight
        )

    def get_o_values(self, measures):
        # measures index order
        # recognized = 2
        # rejected = 1
        # misclassified = 0
        o_values = []
        for i in range(len(measures)):
            o_values.append(
                self.o_func(
                    recognized=measures[i][3],
                    recognized_rejected=measures[i][2],
                    rejected=measures[i][1],
                    misclassified=measures[i][0],
                )
            )
        return o_values

    def fitness_func(self, X, y, mappings):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_size, stratify=y
        )

        return [
            np.sum(
                self.get_o_values(
                    WiSARD(mapping).fit(X_train, y_train).guarisa_measures(X_val, y_val)
                )
            )
            for mapping in mappings
        ]

    def selection(self, population, fitness, k=2):
        min_v = np.min(fitness)
        fitness = (np.array(fitness) - min_v) / (np.max(fitness) - min_v)
        weigths = np.array(fitness) / np.sum(fitness)
        return choices(population, weights=weigths, k=k)

    def run(self, X, y):
        t = 0
        population = self._generate_random_population()
        fitness = self.fitness_func(X, y, population)
        past_mean = np.mean(fitness)
        diff_counter = 0

        while t < self.max_ittr and diff_counter < self.lag:
            t += 1
            offspring_population = []

            for _ in range(self.num_exec):
                choice = random()

                parent_a, parent_b = self.selection(population, fitness)
                if choice < self.theta:
                    child_a, child_b = self.crossover(parent_a, parent_b)
                    offspring_population.append(child_a)
                    offspring_population.append(child_b)
                else:
                    offspring_population.append(self.mutation(parent_a))

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
