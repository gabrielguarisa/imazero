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


def get_default_experiments():
    from imazero.experiments.rand import RandomMonomapping, RandomPolimapping
    from imazero.experiments.wcsp import ConstraintsPolimapping, ConstraintsMonomapping
    from imazero.experiments.pso import AzharParticleSwarmOptimization
    from imazero.wcsp.finders import get_line_finder
    from imazero.experiments.stochasticsearch import (
        AzharStochasticSearch,
        GuarisaStochasticSearch,
    )
    from imazero.experiments.ga import GuarisaGA, GiordanoGA

    return [
        RandomMonomapping(),
        RandomPolimapping(),
        AzharStochasticSearch(0.6, 0.3, 0.1, 1.0),
        GuarisaStochasticSearch(0.5, 0.1, 0.3, 0.1, 1.0),
        ConstraintsPolimapping(get_line_finder()),
        ConstraintsMonomapping(get_line_finder(), "merge"),
        ConstraintsMonomapping(get_line_finder(), "intersection"),
        ConstraintsMonomapping(get_line_finder(), "exclusive"),
        AzharParticleSwarmOptimization(
            recognized_weight=0.5,
            misclassified_weight=0.4,
            rejected_weight=0.1,
            learning_rate=1.0,
            inertia_weight=0.2,
        ),
        GuarisaGA(10),
        GiordanoGA(10)
    ]
