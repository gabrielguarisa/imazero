from imazero.wisard import WiSARD
from .template import TemplateExperiment
from imazero.azhar.stochasticsearch import AzharStochasticSearchMapping
from imazero.guarisa.stochasticsearch import GuarisaStochasticSearchMapping


class AzharStochasticSearch(TemplateExperiment):
    def __init__(
        self,
        recognized_weight,
        misclassified_weight,
        rejected_weight,
        learning_rate,
        save=True,
        folder="results/",
        num_exec=20,
    ):
        self.recognized_weight = recognized_weight
        self.misclassified_weight = misclassified_weight
        self.rejected_weight = rejected_weight
        self.learning_rate = learning_rate
        header_map = {
            "n": int,
            "accuracy": float,
            "recognized_weight": float,
            "misclassified_weight": float,
            "rejected_weight": float,
            "learning_rate": float,
            "generations": int,
        }
        super().__init__(
            "AzharStochasticSearch", header_map, save, folder, num_exec,
        )

    def _prep(self, ds):
        pass

    def _calculate_score(self, ds, tuple_size):
        ss = AzharStochasticSearchMapping(
            tuple_size,
            int(ds.entry_size / tuple_size),
            self.recognized_weight,
            self.misclassified_weight,
            self.rejected_weight,
            self.learning_rate,
        )
        mapping, gen = ss.run(ds.X_train, ds.y_train)

        wsd = WiSARD(mapping).fit(ds.X_train, ds.y_train)
        return {
            "n": tuple_size,
            "accuracy": wsd.score(ds.X_test, ds.y_test),
            "recognized_weight": self.recognized_weight,
            "misclassified_weight": self.misclassified_weight,
            "rejected_weight": self.rejected_weight,
            "learning_rate": self.learning_rate,
            "generations": gen,
        }


class GuarisaStochasticSearch(TemplateExperiment):
    def __init__(
        self,
        recognized_weight,
        recognized_rejected_weight,
        misclassified_weight,
        rejected_weight,
        learning_rate,
        save=True,
        folder="results/",
        num_exec=20,
    ):
        self.recognized_weight = recognized_weight
        self.recognized_rejected_weight = recognized_rejected_weight
        self.misclassified_weight = misclassified_weight
        self.rejected_weight = rejected_weight
        self.learning_rate = learning_rate
        header_map = {
            "n": int,
            "accuracy": float,
            "recognized_weight": float,
            "recognized_rejected_weight": float,
            "misclassified_weight": float,
            "rejected_weight": float,
            "learning_rate": float,
            "generations": int,
        }
        super().__init__(
            "GuarisaStochasticSearch", header_map, save, folder, num_exec,
        )

    def _prep(self, ds):
        pass

    def _calculate_score(self, ds, tuple_size):
        ss = GuarisaStochasticSearchMapping(
            tuple_size,
            int(ds.entry_size / tuple_size),
            self.recognized_weight,
            self.recognized_rejected_weight,
            self.rejected_weight,
            self.misclassified_weight,
            self.learning_rate,
        )
        mapping, gen = ss.run(ds.X_train, ds.y_train)

        wsd = WiSARD(mapping).fit(ds.X_train, ds.y_train)

        return {
            "n": tuple_size,
            "accuracy": wsd.score(ds.X_test, ds.y_test),
            "recognized_weight": self.recognized_weight,
            "recognized_rejected_weight": self.recognized_rejected_weight,
            "misclassified_weight": self.misclassified_weight,
            "rejected_weight": self.rejected_weight,
            "learning_rate": self.learning_rate,
            "generations": gen,
        }
