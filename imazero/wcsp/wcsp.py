import numpy as np
from random import choice, shuffle
from imazero.utils import (
    file_exists,
    load_data,
    save_data,
    get_images_by_label,
    mkdir,
    argsort,
)


def choice_priorities_to_priority_of_choice(choice_priorities):
    return argsort(choice_priorities, True, True)


def calculate_choice_priorities(restrictions, entry_size):
    choice_priorities = [0 for _ in range(entry_size)]

    for r, frequency in restrictions.items():
        choice_priorities[r[0]] += frequency
        choice_priorities[r[1]] += frequency

    return choice_priorities


def function_penalty(chooser, candidate_tuple, restrictions):
    penalties = []
    for addr in candidate_tuple:
        if (addr, chooser) in restrictions:
            penalties.append(restrictions[(addr, chooser)])
        elif (chooser, addr) in restrictions:
            penalties.append(restrictions[(chooser, addr)])

    return penalties


def tuple_selection(chooser, mapping, restrictions, max_tuple_size):
    min_value = 10000000
    best_candidate = None
    total_penalties = 0
    for i in range(len(mapping)):
        if len(mapping[i]) >= max_tuple_size:
            continue

        penalties = function_penalty(chooser, mapping[i], restrictions)
        total_penalty = np.sum(penalties)
        if total_penalty <= min_value:
            change = choice((True, False))
            if total_penalty == min_value:
                if len(penalties) < total_penalties:
                    change = True
            else:
                change = True

            if change:
                min_value = total_penalty
                best_candidate = i
                total_penalties = len(penalties)

    return best_candidate


def create_mapping_by_restrictions(
    restrictions, priority_of_choice, tuple_size, entry_size
):
    number_of_rams = int(entry_size / tuple_size)
    if entry_size % tuple_size > 0:
        raise Exception("Invalid tuple_size value!")

    mapping = [[] for _ in range(number_of_rams)]
    for chooser in priority_of_choice:
        best_tuple = tuple_selection(chooser, mapping, restrictions, tuple_size)
        mapping[best_tuple].append(chooser)

    return mapping


def calculate_restrictions(data, finder):
    restrictions = {}

    for item in data:
        active_addresses = finder.search(item)

        for k in range(len(active_addresses)):
            addresses = active_addresses[k]
            addresses.sort()
            for i in range(len(addresses)):
                for j in range(i, len(addresses)):
                    if i == j:
                        continue

                    r = (addresses[i], addresses[j])
                    if r not in restrictions:
                        restrictions[r] = 0
                    restrictions[r] += 1

    return restrictions


def get_restrictions_and_priorities(X, y, finder, prefix, folder="data/", save=True):
    folder = mkdir(folder)
    restrictions = {}
    priorities = {}

    restrictions_filename = "{}{}_{}_restrictions.pkl".format(folder, prefix, finder)
    priorities_filename = "{}{}_{}_priorities.pkl".format(folder, prefix, finder)

    if file_exists(restrictions_filename) and file_exists(priorities_filename):
        print(
            "Loading: {} and {} files...".format(
                restrictions_filename, priorities_filename
            ),
            end=" ",
        )
        restrictions = load_data(restrictions_filename)
        priorities = load_data(priorities_filename)
        print("Done!")
    else:
        print("Creating restrictions...", end=" ")
        images = get_images_by_label(X, y)
        for label, data in images.items():
            restrictions[label] = calculate_restrictions(data, finder)
            # priorities[label] = calculate_priority_of_choice(
            #     restrictions[label], len(data[0])
            # )
            priorities[label] = calculate_choice_priorities(
                restrictions[label], len(data[0])
            )
        print("Done!")
        if save:
            print(
                "Saving: {} and {} files...".format(
                    restrictions_filename, priorities_filename
                ),
                end=" ",
            )
            save_data(restrictions, restrictions_filename)
            save_data(priorities, priorities_filename)
            print("Done!")

    return restrictions, priorities


def get_merged_restrictions_and_priorities(
    X, y, finder, prefix, folder="data/", save=True
):
    folder = mkdir(folder)
    entry_size = len(X[0])

    restrictions = {}
    priorities = []

    restrictions_filename = "{}{}_{}_merged_restrictions.pkl".format(
        folder, prefix, finder
    )
    priorities_filename = "{}{}_{}_merged_priorities.pkl".format(folder, prefix, finder)

    if file_exists(restrictions_filename) and file_exists(priorities_filename):
        print(
            "Loading: {} and {} files...".format(
                restrictions_filename, priorities_filename
            ),
            end=" ",
        )
        restrictions = load_data(restrictions_filename)
        priorities = load_data(priorities_filename)
        print("Done!")
    else:
        print("Creating restrictions...", end=" ")
        all_restrictions, _ = get_restrictions_and_priorities(X, y, finder, prefix)
        # images = get_images_by_label(X, y)
        labels = list(set(y))

        merged = {}
        for label in labels:
            for key, value in all_restrictions[label].items():
                if key not in merged:
                    merged[key] = 0
                merged[key] += value

        for key in merged.keys():
            new_value = merged[key] / len(labels)
            if new_value > 0:
                restrictions[key] = new_value

        priorities = calculate_choice_priorities(restrictions, entry_size)
        print("Done!")
        if save:
            print(
                "Saving: {} and {} files...".format(
                    restrictions_filename, priorities_filename
                ),
                end=" ",
            )
            save_data(restrictions, restrictions_filename)
            save_data(priorities, priorities_filename)
            print("Done!")

    return restrictions, priorities


def get_intersection_restrictions_and_priorities(
    X, y, finder, prefix, folder="data/", save=True
):
    folder = mkdir(folder)
    entry_size = len(X[0])

    restrictions = {}
    priorities = []

    restrictions_filename = "{}{}_{}_intersection_restrictions.pkl".format(
        folder, prefix, finder
    )
    priorities_filename = "{}{}_{}_intersection_priorities.pkl".format(
        folder, prefix, finder
    )

    if file_exists(restrictions_filename) and file_exists(priorities_filename):
        print(
            "Loading: {} and {} files...".format(
                restrictions_filename, priorities_filename
            ),
            end=" ",
        )
        restrictions = load_data(restrictions_filename)
        priorities = load_data(priorities_filename)
        print("Done!")
    else:
        print("Creating restrictions...", end=" ")
        all_restrictions, _ = get_restrictions_and_priorities(X, y, finder, prefix)
        # images = get_images_by_label(X, y)
        labels = list(set(y))

        for key, value in all_restrictions[labels[0]].items():
            total = 0
            for label, inner_restrictions in all_restrictions.items():
                if key not in inner_restrictions:
                    total = 0
                    break

                total += inner_restrictions[key]

            if total > 0:
                restrictions[key] = total / len(labels)

        priorities = calculate_choice_priorities(restrictions, entry_size)
        print("Done!")
        if save:
            print(
                "Saving: {} and {} files...".format(
                    restrictions_filename, priorities_filename
                ),
                end=" ",
            )
            save_data(restrictions, restrictions_filename)
            save_data(priorities, priorities_filename)
            print("Done!")

    return restrictions, priorities


def get_exlusive_restrictions_and_priorities(
    X, y, finder, prefix, folder="data/", save=True
):
    folder = mkdir(folder)
    entry_size = len(X[0])

    restrictions = {}
    priorities = []

    restrictions_filename = "{}{}_{}_exclusive_restrictions.pkl".format(
        folder, prefix, finder
    )
    priorities_filename = "{}{}_{}_exclusive_priorities.pkl".format(
        folder, prefix, finder
    )

    if file_exists(restrictions_filename) and file_exists(priorities_filename):
        print(
            "Loading: {} and {} files...".format(
                restrictions_filename, priorities_filename
            ),
            end=" ",
        )
        restrictions = load_data(restrictions_filename)
        priorities = load_data(priorities_filename)
        print("Done!")
    else:
        print("Creating restrictions...", end=" ")
        all_restrictions, _ = get_restrictions_and_priorities(X, y, finder, prefix)
        # images = get_images_by_label(X, y)
        labels = list(set(y))

        for key, value in all_restrictions[labels[0]].items():
            equal = True
            for label, inner_restrictions in all_restrictions.items():
                if inner_restrictions.get(key, -1) != value:
                    equal = False
                    break

            if equal:
                restrictions[key] = value

        priorities = calculate_choice_priorities(restrictions, entry_size)
        print("Done!")
        if save:
            print(
                "Saving: {} and {} files...".format(
                    restrictions_filename, priorities_filename
                ),
                end=" ",
            )
            save_data(restrictions, restrictions_filename)
            save_data(priorities, priorities_filename)
            print("Done!")

    return restrictions, priorities
