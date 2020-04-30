import glob
import pandas as pd
import seaborn as sns


def list_experiments(dataset_name, binarization_name, folder="results"):
    prefix = "{}/{}/{}_".format(folder, dataset_name, binarization_name)
    ext = ".csv"
    files = glob.glob("{}*{}".format(prefix, ext))
    return [filename[len(prefix) : -len(ext)] for filename in files]


def get_experimet_scores(
    dataset_name, binarization_name, experiment_name, folder="results"
):
    df = pd.read_csv(
        "{}/{}/{}_{}.csv".format(
            folder, dataset_name, binarization_name, experiment_name
        )
    )
    df = df[["n", "accuracy"]]
    df["experiment"] = experiment_name
    df["n"] = df["n"].astype(int)
    return df


def concat_experiments_scores(
    dataset_name, binarization_name, experiments, folder="results"
):
    data = [
        get_experimet_scores(dataset_name, binarization_name, experiment_name, folder)
        for experiment_name in experiments
    ]

    return pd.concat(data)


def barplot_scores(
    output_file, dataset_name, binarization_name, experiments, results_folder="results"
):
    df = concat_experiments_scores(
        dataset_name, binarization_name, experiments, results_folder
    )
    plot = sns.barplot(x="n", y="accuracy", hue="experiment", data=df)
    fig = plot.get_figure()
    fig.savefig("{}.png".format(output_file))
    fig.clf()
