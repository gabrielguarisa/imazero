import numpy as np
from math import exp
from imazero.wisard import WiSARD
from imazero.utils import get_images_by_label, argsort
from sklearn.model_selection import train_test_split
from random import randint, sample


class GuarisaStochasticSearchMapping(object):
    def __init__(
        self,
        tuple_size,
        final_number_of_tuples,
        recognized_weight,
        recognized_rejected_weight,
        rejected_weight,
        misclassified_weight,
        learning_rate,
        max_ittr=1000,
        lag=10,
        validation_size=0.3,
    ):
        self.tuple_size = tuple_size
        self.final_number_of_tuples = final_number_of_tuples
        self.recognized_weight = recognized_weight
        self.recognized_rejected_weight = recognized_rejected_weight
        self.rejected_weight = rejected_weight
        self.misclassified_weight = misclassified_weight
        self.learning_rate = learning_rate
        self.validation_size = validation_size
        self.max_ittr = max_ittr
        self.lag = lag

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

    def calculate_threshold(self, o_values):
        return np.max(o_values) * (1 - exp(-self.learning_rate / self._t))

    def generate_random_tuples(self, entry_size, num):
        return np.random.randint(int(entry_size), size=(int(num), int(self.tuple_size)))

    def get_best_tuples(self, X, y, mapping):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_size, stratify=y
        )
        o_values = self.get_o_values(
            WiSARD(mapping).fit(X_train, y_train).guarisa_measures(X_val, y_val)
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
        entry_size = len(X[0])
        mature_tuples = []

        past_mean = 0.0
        diff_counter = 0

        while self._t < self.max_ittr and diff_counter < self.lag:
            generation_tuples = [
                *self.generate_random_tuples(entry_size, self.final_number_of_tuples),
                *mature_tuples,
            ]

            mature_tuples, o_values = self.get_best_tuples(
                X, y, generation_tuples
            )

            print(
                "t: {} | dc: {} | Tf: {} | maxO: {} | threshold: {}".format(
                    self._t,
                    diff_counter,
                    len(mature_tuples),
                    np.max(o_values),
                    self.calculate_threshold(o_values),
                ),
                end="\r",
            )

            self._t += 1

            if len(mature_tuples) == self.final_number_of_tuples:
                break
            elif len(mature_tuples) > self.final_number_of_tuples:
                mature_tuples = mature_tuples[: self.final_number_of_tuples]
                break
            else:
                curr_mean = np.mean(o_values)
                if curr_mean == past_mean:
                    diff_counter += 1
                else:
                    diff_counter = 0
                    past_mean = curr_mean
                

        print("")
        return mature_tuples, self._t
