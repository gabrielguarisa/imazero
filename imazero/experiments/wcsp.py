import wisardpkg as wp
from imazero.wisard import RandomWiSARD
from .template import TemplateExperiment
from imazero.wcsp.wcsp import (
    get_restrictions_and_priorities,
    create_mapping_by_restrictions,
    choice_priorities_to_priority_of_choice,
)


class ConstraintsPolimapping(TemplateExperiment):
    def __init__(self, finder, save=True, folder="results/", num_exec=20):
        self.finder = finder
        header_map = {"n": int, "accuracy": float, "finder": str}
        super().__init__(
            "ConstraintsPolimapping", header_map, save, folder, num_exec,
        )

    def _prep(self, ds):
        self.finder.set_shape(ds.shape)
        self.restrictions, self.priorities = get_restrictions_and_priorities(
            ds.X_train, ds.y_train, self.finder, ds.get_name()
        )

    def _calculate_score(self, ds, tuple_size):
        mapping = {}
        for label, choice_priorities in self.priorities.items():
            mapping[label] = create_mapping_by_restrictions(
                self.restrictions[label],
                choice_priorities_to_priority_of_choice(choice_priorities),
                tuple_size,
                ds.entry_size,
            )

        wsd = wp.Wisard(tuple_size, mapping=mapping)
        wsd.train(ds.train)
        return {
            "n": tuple_size,
            "accuracy": wsd.score(ds.test),
            "finder": self.finder.get_name(),
        }
