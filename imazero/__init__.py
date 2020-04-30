def experiment_runner(experiment, dataset_name, binarization_name):
    from imazero.datasets import get_dataset

    ds = get_dataset(dataset_name, binarization_name)
    experiment.run(ds)


def mp_runner(experiments, datasets, binarizations):
    from multiprocessing import Pool, cpu_count

    pool = Pool(cpu_count())

    for experiment in experiments:
        for dataset_name in datasets:
            for binarization_name in binarizations:
                pool.apply_async(
                    experiment_runner,
                    args=(experiment, dataset_name, binarization_name),
                )

    pool.close()
    pool.join()


def mp_runner_config(configurations):
    from multiprocessing import Pool, cpu_count

    pool = Pool(cpu_count())

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
    from imazero.experiments.ga import GuarisaGA, GiordanoGA

    return [
        AzharStochasticSearch(0.6, 0.3, 0.1, 1.0),
        GuarisaStochasticSearch(0.5, 0.1, 0.3, 0.1, 1.0),
        AzharParticleSwarmOptimization(
            recognized_weight=0.5,
            misclassified_weight=0.4,
            rejected_weight=0.1,
            learning_rate=1.0,
            inertia_weight=0.2,
        ),
        GuarisaGA(10),
        GiordanoGA(10),
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
        GroupByMonomapping("mental", "mean"),
        GroupByMonomapping("mental", "std"),
        GroupByMonomapping("mental", "skew"),
        GroupByMonomapping("mental", "kurtosis"),
        GroupByPolimapping("entropy"),
        GroupByPolimapping("mental"),
    ]
