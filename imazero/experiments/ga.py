from imazero.wisard import WiSARD
from .template import TemplateExperiment
from imazero.guarisa.ga import GuarisaGeneticAlgorithm, NewGeneticAlgorithm
from imazero.giordano.ga import GiordanoGeneticAlgorithm
from imazero.utils import valid_tuple_sizes
from contexttimer import Timer


class GuarisaGA(TemplateExperiment):
    def __init__(
        self, population_size, lag=5, save=True, folder="results/", num_exec=20,
    ):
        self.population_size = population_size
        self.lag = 5
        header_map = {
            "n": int,
            "accuracy": float,
            "population_size": int,
            "lag": int,
            "generations": int,
        }
        super().__init__(
            "GuarisaGeneticAlgorithm", header_map, save, folder, num_exec,
        )

    def _prep(self, ds):
        pass

    def _calculate_score(self, ds, tuple_size):
        with Timer(factor=self.factor) as creation:
            ga = GuarisaGeneticAlgorithm(
                tuple_size=tuple_size,
                entry_size=ds.entry_size,
                population_size=self.population_size,
                theta=0.8,
                num_exec=int(self.population_size / 2),
                lag=self.lag,
                max_ittr=100,
                validation_size=0.3,
            )
            mappings, gen = ga.run(ds.X_train, ds.y_train)
            wsd = WiSARD(mappings[0])

        with Timer(factor=self.factor) as training:
            wsd.fit(ds.X_train, ds.y_train)

        with Timer(factor=self.factor) as classification:
            score = wsd.score(ds.X_test, ds.y_test)

        return {
            "n": tuple_size,
            "accuracy": score,
            "creation_time": creation.elapsed,
            "training_time": training.elapsed,
            "classification_time": classification.elapsed,
            "generations": gen,
        }


class GiordanoGA(TemplateExperiment):
    def __init__(
        self,
        population_size,
        theta_r=0.5,
        theta_u=0.2,
        lag=5,
        save=True,
        folder="results/",
        num_exec=20,
    ):
        self.population_size = population_size
        self.lag = 5
        self.theta_r = theta_r
        self.theta_u = theta_u
        header_map = {
            "n": int,
            "accuracy": float,
            "population_size": int,
            "lag": int,
            "generations": int,
            "theta_r": float,
            "theta_u": float,
        }
        super().__init__(
            "GiordanoGeneticAlgorithm", header_map, save, folder, num_exec,
        )

    def _prep(self, ds):
        pass

    def _calculate_score(self, ds, tuple_size):
        with Timer(factor=self.factor) as creation:
            ga = GiordanoGeneticAlgorithm(
                tuple_size=tuple_size,
                entry_size=ds.entry_size,
                population_size=self.population_size,
                theta_r=self.theta_r,
                theta_u=self.theta_u,
                num_exec=int(self.population_size / 2),
                lag=self.lag,
                max_ittr=100,
            )
            mappings, gen = ga.run(ds.X_train, ds.y_train)
            wsd = WiSARD(mappings[0])

        with Timer(factor=self.factor) as training:
            wsd.fit(ds.X_train, ds.y_train)

        with Timer(factor=self.factor) as classification:
            score = wsd.score(ds.X_test, ds.y_test)

        return {
            "n": tuple_size,
            "accuracy": score,
            "creation_time": creation.elapsed,
            "training_time": training.elapsed,
            "classification_time": classification.elapsed,
            "generations": gen,
        }


class NewGA(TemplateExperiment):
    def __init__(
        self,
        population_size,
        recognized_weight,
        recognized_rejected_weight,
        misclassified_weight,
        rejected_weight,
        lag=5,
        save=True,
        folder="results/",
        num_exec=20,
    ):
        self.population_size = population_size
        self.lag = 5
        self.recognized_weight = recognized_weight
        self.recognized_rejected_weight = recognized_rejected_weight
        self.rejected_weight = rejected_weight
        self.misclassified_weight = misclassified_weight

        header_map = {
            "n": int,
            "accuracy": float,
            "population_size": int,
            "lag": int,
            "generations": int,
        }
        super().__init__(
            "NewGeneticAlgorithm", header_map, save, folder, num_exec,
        )

    def _prep(self, ds):
        pass

    def _calculate_score(self, ds, tuple_size):
        with Timer(factor=self.factor) as creation:
            ga = NewGeneticAlgorithm(
                tuple_size=tuple_size,
                entry_size=ds.entry_size,
                population_size=self.population_size,
                theta=0.8,
                num_exec=int(self.population_size / 2),
                lag=self.lag,
                max_ittr=100,
                validation_size=0.3,
                recognized_weight=self.recognized_weight,
                recognized_rejected_weight=self.recognized_rejected_weight,
                misclassified_weight=self.misclassified_weight,
                rejected_weight=self.rejected_weight,
            )
            mappings, gen = ga.run(ds.X_train, ds.y_train)
            wsd = WiSARD(mappings[0])

        with Timer(factor=self.factor) as training:
            wsd.fit(ds.X_train, ds.y_train)

        with Timer(factor=self.factor) as classification:
            score = wsd.score(ds.X_test, ds.y_test)

        return {
            "n": tuple_size,
            "accuracy": score,
            "creation_time": creation.elapsed,
            "training_time": training.elapsed,
            "classification_time": classification.elapsed,
            "generations": gen,
        }
