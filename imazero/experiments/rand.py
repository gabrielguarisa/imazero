import wisardpkg as wp
from imazero.wisard import RandomWiSARD
from .template import TemplateExperiment

class RandomPolimapping(TemplateExperiment):
    def __init__(self, save=True, folder="results/", num_exec=20):
        header_map = {"n": int, "accuracy": float}
        super().__init__("RandomPolimapping", header_map, save, folder, num_exec)

    def _calculate_score(self, ds, tuple_size):
        wsd = wp.Wisard(tuple_size)
        wsd.train(ds.train)
        return {"n": tuple_size, "accuracy": wsd.score(ds.test)}


class RandomMonomapping(TemplateExperiment):
    def __init__(self, save=True, folder="results/", num_exec=20):
        header_map = {"n": int, "accuracy": float}
        super().__init__("RandomMonomapping", header_map, save, folder, num_exec)

    def _calculate_score(self, ds, tuple_size):
        wsd = RandomWiSARD(tuple_size, ds.entry_size).fit(ds.X_train, ds.y_train)
        return {"n": tuple_size, "accuracy": wsd.score(ds.X_test, ds.y_test)}
