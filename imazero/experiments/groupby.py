import wisardpkg as wp
from imazero.wisard import WiSARD
from .template import TemplateExperiment
from imazero.utils import get_mental_images, get_entropy_images, argsort
import numpy as np
from scipy.stats import skew, kurtosis


class GroupByPolimapping(TemplateExperiment):
    def __init__(self, metric, save=True, folder="results/", num_exec=20):
        header_map = {"n": int, "accuracy": float}
        self._metric = metric.lower()
        super().__init__(
            "GroupBy{}Polimapping".format(metric.title()),
            header_map,
            save,
            folder,
            num_exec,
        )

    def _prep(self, ds):
        if self._metric == "entropy":
            self._metric_images = get_entropy_images(ds.X_train, ds.y_train)
        elif self._metric == "mental":
            self._metric_images = get_mental_images(ds.X_train, ds.y_train)
        else:
            raise Exception("Metric not found!")

    def _calculate_score(self, ds, tuple_size):
        mapping = {}

        for label, img in self._metric_images.items():
            mapping[str(label)] = np.reshape(argsort(img), (-1, tuple_size))

        wsd = wp.Wisard(tuple_size, mapping=mapping)
        wsd.train(ds.train)
        return {"n": tuple_size, "accuracy": wsd.score(ds.test)}


class GroupByMonomapping(TemplateExperiment):
    def __init__(self, metric, func, save=True, folder="results/", num_exec=20):
        header_map = {"n": int, "accuracy": float}
        self._metric = metric.lower()
        self._func = func.lower()
        super().__init__(
            "GroupBy{}{}Monomapping".format(metric.title(), func.title()),
            header_map,
            save,
            folder,
            num_exec,
        )

    def _prep(self, ds):
        if self._metric == "entropy":
            metric_images = get_entropy_images(ds.X_train, ds.y_train)
        elif self._metric == "mental":
            metric_images = get_mental_images(ds.X_train, ds.y_train)
        else:
            raise Exception("Metric not found!")

        images = list(metric_images.values())
        if self._func == "mean":
            self._metric_image = np.array(np.mean(images, axis=0), dtype="float16")
        elif self._func == "std":
            self._metric_image = np.array(np.std(images, axis=0), dtype="float16")
        elif self._func == "skew":
            self._metric_image = np.array(skew(images, axis=0), dtype="float16")
        elif self._func == "kurtosis":
            self._metric_image = np.array(kurtosis(images, axis=0), dtype="float16")
        else:
            raise Exception("Invalid function!")

    def _calculate_score(self, ds, tuple_size):
        mapping = np.reshape(argsort(self._metric_image), (-1, tuple_size))
        wsd = WiSARD(mapping).fit(ds.X_train, ds.y_train)
        return {"n": tuple_size, "accuracy": wsd.score(ds.X_test, ds.y_test)}
