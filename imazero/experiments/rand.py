import wisardpkg as wp
from imazero.wisard import RandomWiSARD, RandomPolimappingWiSARD
from imazero.utils import random_mapping
from .template import TemplateExperiment
from contexttimer import Timer

class RandomPolimapping(TemplateExperiment):
    def __init__(self, save=True, folder="results/", num_exec=20):
        header_map = {"n": int, "accuracy": float}
        super().__init__("RandomPolimapping", header_map, save, folder, num_exec)

    def _prep(self, ds):
        pass

    def _calculate_score(self, ds, tuple_size):
        with Timer(factor=self.factor) as creation:
            wsd = RandomPolimappingWiSARD(tuple_size, ds.num_classes, ds.entry_size)

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
        }


class RandomMonomapping(TemplateExperiment):
    def __init__(self, save=True, folder="results/", num_exec=20):
        header_map = {"n": int, "accuracy": float}
        super().__init__("RandomMonomapping", header_map, save, folder, num_exec)

    def _prep(self, ds):
        pass

    def _calculate_score(self, ds, tuple_size):
        with Timer(factor=self.factor) as creation:
            wsd = RandomWiSARD(tuple_size, ds.entry_size)

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
        }

