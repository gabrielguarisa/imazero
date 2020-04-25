from imazero.wisard import WiSARD
from .template import TemplateExperiment
from imazero.azhar.pso import AzharParticleSwarmMapping


class AzharParticleSwarmOptimization(TemplateExperiment):
    def __init__(
        self,
        recognized_weight,
        misclassified_weight,
        rejected_weight,
        learning_rate,
        inertia_weight,
        save=True,
        folder="results/",
        num_exec=20,
    ):
        self.recognized_weight = recognized_weight
        self.misclassified_weight = misclassified_weight
        self.rejected_weight = rejected_weight
        self.learning_rate = learning_rate
        self.inertia_weight = inertia_weight
        header_map = {
            "n": int,
            "accuracy": float,
            "recognized_weight": float,
            "misclassified_weight": float,
            "rejected_weight": float,
            "learning_rate": float,
            "inertia_weight": float,
            "generations": int,
        }
        super().__init__(
            "AzharParticleSwarmOptimization", header_map, save, folder, num_exec,
        )

    def _prep(self, ds):
        pass

    def _calculate_score(self, ds, tuple_size):
        ss = AzharParticleSwarmMapping(
            tuple_size=tuple_size,
            final_number_of_tuples=int(ds.entry_size / tuple_size),
            inertia_weight=self.inertia_weight,
            recognized_weight=self.recognized_weight,
            misclassified_weight=self.misclassified_weight,
            rejected_weight=self.rejected_weight,
            learning_rate=self.learning_rate,
            criticality_limit=3,
        )
        mapping, gen = ss.create_mapping(ds)

        wsd = WiSARD(mapping).fit(ds.X_train, ds.y_train)
        return {
            "n": tuple_size,
            "accuracy": wsd.score(ds.X_test, ds.y_test),
            "recognized_weight": self.recognized_weight,
            "misclassified_weight": self.misclassified_weight,
            "rejected_weight": self.rejected_weight,
            "learning_rate": self.learning_rate,
            "inertia_weight": self.inertia_weight,
            "generations": gen,
        }
