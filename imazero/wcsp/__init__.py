from imazero.wcsp.wcsp import (
    # get_restrictions_and_priorities,
    create_mapping_by_restrictions,
    get_merged_restrictions_and_priorities,
    get_intersection_restrictions_and_priorities,
    get_exlusive_restrictions_and_priorities,
    choice_priorities_to_priority_of_choice,
)
import numpy as np


class ConstraintsMonomapping:
    def __init__(self, X, y, finder, operation_name, dataset_name):
        self.operation_func = None
        if operation_name.lower() == "merge":
            operation_func = get_merged_restrictions_and_priorities
        elif operation_name.lower() == "intersection":
            operation_func = get_intersection_restrictions_and_priorities
        elif operation_name.lower() == "exclusive":
            operation_func = get_exlusive_restrictions_and_priorities
        else:
            raise Exception("Invalid operation!")
        self.restrictions, self.choice_priorities = operation_func(
            X, y, finder, dataset_name
        )

    def create(self, tuple_size, entry_size):
        return create_mapping_by_restrictions(
            self.restrictions,
            choice_priorities_to_priority_of_choice(self.choice_priorities),
            tuple_size,
            entry_size,
        )
