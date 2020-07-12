import argparse
from imazero import (
    mp_runner,
    runner,
    get_random_experiments,
    get_stochastic_experiments,
    get_wcsp_line_experiments,
    get_wcsp_sparse_experiments,
    get_group_by_experiments,
    plots,
)
from multiprocessing import cpu_count

configurations = {
    "mnist": {
        "binarizations": ["mt", "ot", "lt"],
        "experiments": [
            *get_group_by_experiments(),
            *get_random_experiments(),
            *get_stochastic_experiments(),
            *get_wcsp_line_experiments(),
        ],
    },
    "fashion": {
        "binarizations": ["mt", "ot", "lt"],
        "experiments": [
            *get_group_by_experiments(),
            *get_random_experiments(),
            *get_stochastic_experiments(),
            *get_wcsp_line_experiments(),
        ],
    },
    "cifar10": {
        "binarizations": ["mt", "ot", "lt"],
        "experiments": [
            *get_group_by_experiments(),
            *get_random_experiments(),
            *get_stochastic_experiments(),
            *get_wcsp_line_experiments(),
        ],
    },
    "ckp": {
        "binarizations": ["mt", "ot", "lt"],
        "experiments": [
            *get_group_by_experiments(),
            *get_random_experiments(),
            *get_stochastic_experiments(),
            *get_wcsp_line_experiments(),
        ],
    },
    "imdb": {
        "binarizations": ["th"],
        "experiments": [
            *get_group_by_experiments(),
            *get_random_experiments(),
            *get_stochastic_experiments(),
            *get_wcsp_sparse_experiments(),
        ],
    },
    "movielens": {
        "binarizations": ["th"],
        "experiments": [
            *get_group_by_experiments(),
            *get_random_experiments(),
            *get_stochastic_experiments(),
            *get_wcsp_sparse_experiments(),
        ],
    },
}

experiments_desc = {
    "wcsp": [
        "ConstraintsMonomappingMerge",
        "ConstraintsMonomappingExclusive",
        "ConstraintsMonomappingIntersection",
        "ConstraintsPolimapping",
        "RandomMonomapping",
        "RandomPolimapping",
    ],
    "stochastic": [
        "AzharParticleSwarmOptimization",
        "AzharStochasticSearch",
        "GiordanoGeneticAlgorithm",
        "GuarisaStochasticSearch",
        "GuarisaGeneticAlgorithm",
        "RandomMonomapping",
        "RandomPolimapping",
    ],
    "groupby": [
        "GroupByEntropyMeanMonomapping",
        "GroupByEntropyStdMonomapping",
        "GroupByEntropySkewMonomapping",
        "GroupByEntropyKurtosisMonomapping",
        "GroupByMentalMeanMonomapping",
        "GroupByMentalStdMonomapping",
        "GroupByMentalSkewMonomapping",
        "GroupByMentalKurtosisMonomapping",
        "GroupByEntropyPolimapping",
        "GroupByMentalPolimapping",
    ],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument(
        "datasets", metavar="N", type=str, nargs="+", help="dataset names"
    )

    parser.add_argument(
        "--metric", action="store_true", dest="metric", help="Plot metric images.",
    )

    parser.add_argument(
        "--plot", action="store_true", dest="plot", help="Plot accuracy images.",
    )

    parser.add_argument(
        "--sequential",
        action="store_true",
        dest="sequential",
        help="Run without paralelism.",
    )

    parser.add_argument(
        "-n",
        "--num-procs",
        action="store",
        dest="num_procs",
        default=cpu_count(),
        type=int,
        help="Number of processes.",
    )

    args = parser.parse_args()

    if args.metric or args.plot:
        if args.metric:
            for dataset_name in args.datasets:
                for binarization_name in configurations[dataset_name]["binarizations"]:
                    plots.plot_metric_images(dataset_name, binarization_name)
                    plots.plot_metric_images(
                        dataset_name, binarization_name, entropy=True
                    )
        else:
            for dataset_name in args.datasets:
                for binarization_name in configurations[dataset_name]["binarizations"]:
                    for desc, experiment_names in experiments_desc.items():
                        plots.barplot_scores(
                            desc, dataset_name, binarization_name, experiment_names,
                        )
    else:
        if args.sequential:
            for dataset_name in args.datasets:
                runner(
                    experiments=configurations[dataset_name]["experiments"],
                    datasets=[dataset_name],
                    binarizations=configurations[dataset_name]["binarizations"],
                )
        else:
            for dataset_name in args.datasets:
                mp_runner(
                    experiments=configurations[dataset_name]["experiments"],
                    datasets=[dataset_name],
                    binarizations=configurations[dataset_name]["binarizations"],
                    num_procs=args.num_procs,
                )
