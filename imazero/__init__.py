import datasets


def experiment_runner(experiment, dataset_name, binarization_name):
    from imazero.datasets import get_dataset

    ds = get_dataset(dataset_name, binarization_name)
    experiment.run(ds)


def runner(experiments, datasets, binarizations):
    for experiment in experiments:
        for dataset_name in datasets:
            for binarization_name in binarizations:
                experiment_runner(experiment, dataset_name, binarization_name)


def mp_runner(experiments, datasets, binarizations, num_procs):
    from multiprocessing import Pool

    pool = Pool(num_procs)

    for experiment in experiments:
        for dataset_name in datasets:
            for binarization_name in binarizations:
                pool.apply_async(
                    experiment_runner,
                    args=(experiment, dataset_name, binarization_name),
                )

    pool.close()
    pool.join()


def mp_runner_config(configurations, num_procs):
    from multiprocessing import Pool

    pool = Pool(num_procs)

    for dataset_name, config in configurations.items():
        for experiment in config["experiments"]:
            for binarization_name in config["binarizations"]:
                pool.apply_async(
                    experiment_runner,
                    args=(experiment, dataset_name, binarization_name),
                )

    pool.close()
    pool.join()


def get_stochastic_experiments():
    from imazero.experiments.pso import AzharParticleSwarmOptimization
    from imazero.experiments.stochasticsearch import (
        AzharStochasticSearch,
        GuarisaStochasticSearch,
    )
    from imazero.experiments.ga import GuarisaGA, GiordanoGA, NewGA

    return [
        AzharStochasticSearch(
            recognized_weight=39.0,
            misclassified_weight=-0.5,
            rejected_weight=-1.0,
            learning_rate=100,
        ),
        GuarisaStochasticSearch(
            recognized_weight=39.0,
            recognized_rejected_weight=19.5,
            misclassified_weight=-0.5,
            rejected_weight=-1.0,
            learning_rate=100,
        ),
        AzharParticleSwarmOptimization(
            recognized_weight=39.0,
            misclassified_weight=-0.5,
            rejected_weight=-1.0,
            learning_rate=100,
            inertia_weight=0.2,
        ),
        GiordanoGA(10),
        # NewGA(
        #     10,
        #     recognized_weight=39.0,
        #     recognized_rejected_weight=19.5,
        #     misclassified_weight=-0.5,
        #     rejected_weight=-1.0,
        # ),
        GuarisaGA(10),
    ]


def get_random_experiments():
    from imazero.experiments.rand import RandomMonomapping, RandomPolimapping

    return [
        RandomMonomapping(),
        RandomPolimapping(),
    ]


def get_wcsp_line_experiments():
    from imazero.experiments.wcsp import ConstraintsPolimapping, ConstraintsMonomapping
    from imazero.wcsp.finders import get_line_finder

    return [
        ConstraintsPolimapping(get_line_finder()),
        ConstraintsMonomapping(get_line_finder(), "merge"),
        ConstraintsMonomapping(get_line_finder(), "intersection"),
        ConstraintsMonomapping(get_line_finder(), "exclusive"),
    ]


def get_wcsp_sparse_experiments():
    from imazero.experiments.wcsp import ConstraintsPolimapping, ConstraintsMonomapping
    from imazero.wcsp.finders import OnesFinder

    return [
        ConstraintsPolimapping(OnesFinder()),
        ConstraintsMonomapping(OnesFinder(), "merge"),
        ConstraintsMonomapping(OnesFinder(), "intersection"),
        ConstraintsMonomapping(OnesFinder(), "exclusive"),
    ]


def get_group_by_experiments():
    from imazero.experiments.groupby import GroupByMonomapping, GroupByPolimapping

    return [
        GroupByMonomapping("entropy", "mean"),
        GroupByMonomapping("entropy", "std"),
        GroupByMonomapping("entropy", "skew"),
        GroupByMonomapping("entropy", "kurtosis"),
        GroupByPolimapping("entropy"),
        # GroupByMonomapping("mental", "mean"),
        # GroupByMonomapping("mental", "std"),
        # GroupByMonomapping("mental", "skew"),
        # GroupByMonomapping("mental", "kurtosis"),
        # GroupByPolimapping("mental"),
    ]
