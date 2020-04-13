from imazero.utils import valid_tuple_sizes, mkdir, file_exists
from abc import ABC, abstractmethod
import pandas as pd


class TemplateExperiment(ABC):
    def __init__(
        self, experiment_name, header_map, save=True, folder="results/", num_exec=20
    ):
        self.experiment_name = experiment_name
        self.header_map = header_map
        self.save = save
        self.folder = mkdir(folder)
        self.num_exec = num_exec
        super().__init__()

    def get_folder(self, dataset_name):
        folder_name = "{}/{}/".format(self.folder, dataset_name)
        return mkdir(folder_name)

    def get_filename(self, dataset_name, binarization_name):
        return "{}{}_{}.csv".format(
            self.get_folder(dataset_name), binarization_name, self.experiment_name
        )

    def load(self, filename):
        if file_exists(filename):
            return pd.read_csv(filename)

        df = pd.DataFrame(columns=list(self.header_map.keys()))
        return df.astype(self.header_map)

    @abstractmethod
    def _calculate_score(self, ds, tuple_size):
        pass

    def run(self, ds):
        print(
            "=== INFO ===\nExperimento: {}\nDataset: {}".format(self.experiment_name, ds.get_name())
        )
        filename = self.get_filename(ds.dataset_name, ds.binarization_name)
        df = self.load(filename)
        tuple_sizes = valid_tuple_sizes(ds.entry_size)
        print("Tuple sizes: ", tuple_sizes)
        for tuple_size in tuple_sizes:
            print("-- {}".format(tuple_size), end="\r")
            nec_exec = self.num_exec - len(df.loc[df["n"] == tuple_size])
            for _ in range(nec_exec):
                df = df.append(
                    self._calculate_score(ds, tuple_size), ignore_index=True,
                )
            print("-- {}\tOK!".format(tuple_size))

            if self.save:
                df.to_csv(filename, index=False)
        print("=============")
        return df