import numpy as np
from math import exp
from imazero.wisard import WiSARD
from imazero.utils import get_images_by_label, argsort
from sklearn.model_selection import train_test_split
from random import randint, sample


class AzharStochasticSearchMapping(object):
    def __init__(
        self,
        tuple_size,
        final_number_of_tuples,
        recognized_weight,
        misclassified_weight,
        rejected_weight,
        learning_rate,
        validation_size=0.3,
    ):
        self.tuple_size = tuple_size
        self.final_number_of_tuples = final_number_of_tuples
        self.recognized_weight = recognized_weight
        self.misclassified_weight = misclassified_weight
        self.rejected_weight = rejected_weight
        self.learning_rate = learning_rate
        self.validation_size = validation_size

    def o_func(self, recognized, misclassified, rejected):
        return (
            recognized * self.recognized_weight
            - misclassified * self.misclassified_weight
            - rejected * self.rejected_weight
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
                    recognized=measures[i][2],
                    rejected=measures[i][1],
                    misclassified=measures[i][0],
                )
            )
        return o_values

    def calculate_threshold(self, o_values):
        return np.max(o_values) * (1 - exp(-self.learning_rate / self._t))

    def generate_random_tuples(self, entry_size, num):
        return np.random.randint(int(entry_size), size=(int(num), int(self.tuple_size)))

    def get_best_tuples(self, X, y, y_true, mapping):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_size, stratify=y
        )
        splitted = get_images_by_label(X_val, y_val)
        o_values = self.get_o_values(
            WiSARD(mapping)
            .fit(X_train, y_train)
            .azhar_measures(splitted[y_true], [y_true] * len(splitted[y_true]))
        )
        threshold = self.calculate_threshold(o_values)
        valid_tuples = []
        indexes = argsort(o_values)
        for i in indexes:
            if o_values[i] > threshold:
                valid_tuples.append(mapping[i])

        return valid_tuples, o_values

    def run(self, X, y):
        self._t = 1
        labels = list(set(y))
        num_labels = len(labels)
        entry_size = len(X[0])
        mature_tuples = []
        class_tuples = []
        i = 0
        tuples_per_class = int(self.final_number_of_tuples / num_labels)
        num_tuples = [tuples_per_class for _ in range(num_labels)]
        remainder = self.final_number_of_tuples % num_labels
        if remainder > 0:
            indexes = sample(labels, remainder)
            for j in indexes:
                num_tuples[j] += 1

        gen = 1
        while i < num_labels:
            generation_tuples = [
                *self.generate_random_tuples(entry_size, num_tuples[i]),
                *class_tuples,
            ]
            class_tuples, o_values = self.get_best_tuples(
                X, y, labels[i], generation_tuples
            )
            print(
                "gen: {} | t: {} | i: {} | Tf: {} | maxO: {} | threshold: {}".format(
                    gen,
                    self._t,
                    i,
                    len(class_tuples),
                    np.max(o_values),
                    self.calculate_threshold(o_values),
                ),
                end="\r",
            )

            self._t += 1
            gen += 1
            if len(class_tuples) >= tuples_per_class  or self._t > 1000:
                mature_tuples = [*mature_tuples, *class_tuples[:tuples_per_class]]
                i += 1
                class_tuples = []
                self._t = 1
        print("")
        return mature_tuples, gen
