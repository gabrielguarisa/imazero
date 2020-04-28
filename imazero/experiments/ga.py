from imazero.wisard import WiSARD
from .template import TemplateExperiment
from imazero.guarisa.ga import GuarisaGeneticAlgorithm
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
        pass

    def run(self, ds):
        print(
            "=== INFO ===\nExperimento: {}\nDataset: {}".format(
                self.experiment_name, ds.get_name()
            )
        )
        filename = self.get_filename(ds.dataset_name, ds.binarization_name)
        df = self.load(filename)
        tuple_sizes = valid_tuple_sizes(ds.entry_size)
        print("Tuple sizes: ", tuple_sizes)
        for tuple_size in tuple_sizes:
            nec_exec = self.num_exec - len(df.loc[df["n"] == tuple_size])
            ga = GuarisaGeneticAlgorithm(
                tuple_size,
                ds.entry_size,
                population_size=self.population_size,
                num_exec=nec_exec,
                lag=self.lag,
            )
            mappings, gen = ga.run(ds.X_train, ds.y_train)
            for mapping in mappings:
                score = (
                    WiSARD(mapping)
                    .fit(ds.X_train, ds.y_train)
                    .score(ds.X_test, ds.y_test)
                )
                df = df.append(
                    {
                        "n": tuple_size,
                        "accuracy": score,
                        "population_size": self.population_size,
                        "lag": self.lag,
                        "generations": gen,
                    },
                    ignore_index=True,
                )
            print("-- {}\tOK!    ".format(tuple_size))

            if self.save:
                df.to_csv(filename, index=False)
        print("=============")
        return df
