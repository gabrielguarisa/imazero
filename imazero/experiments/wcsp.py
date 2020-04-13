import wisardpkg as wp
from imazero.wisard import WiSARD
from .template import TemplateExperiment
from imazero.wcsp.wcsp import (
    get_restrictions_and_priorities,
    create_mapping_by_restrictions,
    choice_priorities_to_priority_of_choice,
    get_merged_restrictions_and_priorities,
    get_intersection_restrictions_and_priorities,
    get_exlusive_restrictions_and_priorities,
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
            mapping[str(label)] = create_mapping_by_restrictions(
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


class ConstraintsMonomapping(TemplateExperiment):
    def __init__(
        self, finder, operation_name, save=True, folder="results/", num_exec=20
    ):
        self.finder = finder
        header_map = {"n": int, "accuracy": float, "finder": str}
        super().__init__(
            "ConstraintsMonomapping{}".format(operation_name.title()),
            header_map,
            save,
            folder,
            num_exec,
        )
        self.operation_func = None
        if operation_name.lower() == "merge":
            self.operation_func = get_merged_restrictions_and_priorities
        elif operation_name.lower() == "intersection":
            self.operation_func = get_intersection_restrictions_and_priorities
        elif operation_name.lower() == "exclusive":
            self.operation_func = get_exlusive_restrictions_and_priorities
        else:
            raise Exception("Invalid operation!")

    def _prep(self, ds):
        self.finder.set_shape(ds.shape)
        self.restrictions, self.priorities = self.operation_func(
            ds.X_train, ds.y_train, self.finder, ds.get_name()
        )

    def _calculate_score(self, ds, tuple_size):
        wsd = WiSARD(
            create_mapping_by_restrictions(
                self.restrictions,
                choice_priorities_to_priority_of_choice(self.priorities),
                tuple_size,
                ds.entry_size,
            )
        ).fit(ds.X_train, ds.y_train)
        return {
            "n": tuple_size,
            "accuracy": wsd.score(ds.X_test, ds.y_test),
            "finder": self.finder.get_name(),
        }
