import numpy as np
from math import exp
from imazero.wisard import WiSARD
from imazero.utils import get_images_by_label, argsort
from sklearn.model_selection import train_test_split
from random import randint, sample
from .stochasticsearch import AzharStochasticSearchMapping
from .particle import Particle


class AzharParticleSwarmMapping(AzharStochasticSearchMapping):
    def __init__(
        self,
        tuple_size,
        num_particles,
        inertia_weight,
        final_number_of_tuples,
        recognized_weight,
        misclassified_weight,
        rejected_weight,
        learning_rate,
        local_acceleration=1.0,
        global_acceleration=1.0,
        validation_size=0.3,
        criticality_limit=None,
    ):
        self.local_acceleration = local_acceleration
        self.global_acceleration = global_acceleration
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight
        self.criticality_limit = criticality_limit

        super().__init__(
            tuple_size,
            final_number_of_tuples,
            recognized_weight,
            misclassified_weight,
            recognized_weight,
            learning_rate,
            validation_size,
        )

    def generate_random_particles(self, entry_size):
        return [
            Particle(
                initial_position=np.random.randint(
                    entry_size - 1, size=self.tuple_size
                ),
                initial_velocity=np.random.uniform(-1.0, 1.0, size=self.tuple_size),
                lower_bound=0,
                upper_bound=entry_size - 1,
                local_acceleration=self.local_acceleration,
                global_acceleration=self.global_acceleration,
            )
            for _ in range(self.num_particles)
        ]

    def particles_to_mapping(self, particles):
        return [p.get_position() for p in particles]

    def get_best_tuples(self, X, y, y_true, particles):
        mapping = self.particles_to_mapping(particles)
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
            particles[i].update_best_score(o_values[i])
            if o_values[i] > threshold:
                valid_tuples.append(mapping[i])

        return particles[indexes[0]].get_position(), valid_tuples, particles

    def create_mapping(self, ds):
        self._t = 1
        labels = list(set(ds.y_train))
        num_labels = len(labels)
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

        cl_min = round((self.num_particles * self.tuple_size) / ds.entry_size)
        rows, cols = ds.shape
        if self.criticality_limit == None:
            self.criticality_limit = cl_min
        elif self.criticality_limit < cl_min:
            raise Exception("Invalid criticality limit!")

       
        while i < num_labels:
            particles = self.generate_random_particles(ds.entry_size)
            best_position, _, particles = self.get_best_tuples(
                ds.X_train, ds.y_train, labels[i], particles
            )

            criticality = np.zeros(ds.entry_size, dtype=int)

            for j in range(self.num_particles):
                for d in range(self.tuple_size):

                    particles[j].update_velocity(self.inertia_weight, best_position, d)

                    particles[j].update_position(d)

                    criticality[particles[j].get_position(d)] += 1

                    while (
                        criticality[particles[j].get_position(d)]
                        > self.criticality_limit
                    ):
                        criticality[particles[j].get_position(d)] -= 1
                        particles[j].disperse(d, rows, cols)
                        criticality[particles[j].get_position(d)] += 1
                        print(
                            j,
                            d,
                            particles[j].get_position(d),
                            criticality[particles[j].get_position(d)],
                            end="\r",
                        )

            print(
                "gen: {} | t: {} | i: {} | Tf: {}".format(
                    gen, self._t, i, len(class_tuples)
                ),
                end="\r",
            )
            best_position, class_tuples, particles = self.get_best_tuples(
                ds.X_train, ds.y_train, labels[i], particles
            )
            self._t += 1
            gen += 1
            if len(class_tuples) >= tuples_per_class:
                mature_tuples = [*mature_tuples, *class_tuples[:tuples_per_class]]
                i += 1
                class_tuples = []
                self._t = 1
        print("")
        return mature_tuples
