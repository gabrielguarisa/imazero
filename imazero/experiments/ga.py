from imazero.wisard import WiSARD
from .template import TemplateExperiment
from imazero.guarisa.ga import GuarisaGeneticAlgorithm
from imazero.giordano.ga import GiordanoGeneticAlgorithm
from imazero.utils import valid_tuple_sizes


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
        ga = GuarisaGeneticAlgorithm(
            tuple_size,
            ds.entry_size,
            population_size=self.population_size,
            num_exec=int(self.population_size / 2),
            lag=self.lag, recognized_weight=2, recognized_rejected_weight=1, rejected_weight=1, misclassified_weight=2
        )
        mappings, gen = ga.run(ds.X_train, ds.y_train)

        score = (
            WiSARD(mappings[0]).fit(ds.X_train, ds.y_train).score(ds.X_test, ds.y_test)
        )
        return {
            "n": tuple_size,
            "accuracy": score,
            "population_size": self.population_size,
            "lag": self.lag,
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
        ga = GiordanoGeneticAlgorithm(
            tuple_size,
            ds.entry_size,
            self.population_size,
            self.theta_r,
            self.theta_u,
            num_exec=int(self.population_size / 2),
        )
        mappings, gen = ga.run(ds.X_train, ds.y_train)

        score = (
            WiSARD(mappings[0]).fit(ds.X_train, ds.y_train).score(ds.X_test, ds.y_test)
        )
        return {
            "n": tuple_size,
            "accuracy": score,
            "population_size": self.population_size,
            "lag": self.lag,
            "generations": gen,
            "theta_r": self.theta_r,
            "theta_u": self.theta_u,
        }
