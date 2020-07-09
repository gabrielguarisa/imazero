from imazero.utils import load_data
from imazero.datasets import Dataset
import numpy as np

raw_folder = "datasets/raw/"
binarization = "sv"

train = load_data("{}ckp_train.pkl".format(raw_folder))
test = load_data("{}ckp_test.pkl".format(raw_folder))

num_classes = len(np.unique(train["labels"]))
entry_size = len(train["images"][0].ravel())

print(num_classes)
print(entry_size)

Dataset.binarize(
    train["images"],
    train["labels"],
    test["images"],
    test["labels"],
    dataset_name="ckp",
    binarization_name=binarization,
    num_classes=num_classes,
    entry_size=entry_size,
    shape=(100, 100),
).save("datasets/data/")

