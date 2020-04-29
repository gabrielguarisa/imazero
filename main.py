import argparse
from imazero import (
    mp_runner,
    get_random_experiments,
    get_stochastic_experiments,
    get_wcsp_line_experiments,
    get_wcsp_sparse_experiments,
)

configurations = {
    "mnist": {
        "binarizations": ["mt", "ot"],
        "experiments": [
            *get_random_experiments(),
            *get_stochastic_experiments(),
            *get_wcsp_line_experiments(),
        ],
    },
    "fashion": {
        "binarizations": ["mt", "ot"],
        "experiments": [
            *get_random_experiments(),
            *get_stochastic_experiments(),
            *get_wcsp_line_experiments(),
        ],
    },
    "cifar10": {
        "binarizations": ["mt", "ot"],
        "experiments": [
            *get_random_experiments(),
            *get_stochastic_experiments(),
            *get_wcsp_line_experiments(),
        ],
    },
    "ckp": {
        "binarizations": ["mt", "ot"],
        "experiments": [
            *get_random_experiments(),
            *get_stochastic_experiments(),
            *get_wcsp_line_experiments(),
        ],
    },
    "imdb": {
        "binarizations": ["mt"],
        "experiments": [
            *get_random_experiments(),
            *get_stochastic_experiments(),
            *get_wcsp_sparse_experiments(),
        ],
    },
    "movielens": {
        "binarizations": ["th"],
        "experiments": [
            *get_random_experiments(),
            *get_stochastic_experiments(),
            *get_wcsp_sparse_experiments(),
        ],
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument(
        "datasets", metavar="N", type=str, nargs="+", help="dataset names"
    )

    args = parser.parse_args()

    for dataset_name in args.datasets:
        mp_runner(
            experiments=configurations[dataset_name]["experiments"],
            datasets=[dataset_name],
            binarizations=configurations[dataset_name]["binarizations"],
        )
