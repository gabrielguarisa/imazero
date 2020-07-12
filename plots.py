import numpy as np
from skimage import io
import wisardpkg as wp
from imazero.datasets import get_dataset


def save_binary_img(arr, filename):
    io.imsave(filename, arr.astype(np.uint16))


def get_ds_example(dataset_name, binarization_name, shape, index=0):
    ds = get_dataset(dataset_name, binarization_name)

    return np.array(ds.train[index].list()).reshape(shape) * 255


dataset_name = "ckp"
shape = (100, 100)

for binarization_name in ["lt", "mt", "ot"]:
    for i in range(120):
        save_binary_img(
            get_ds_example(dataset_name, binarization_name, shape, i),
            "tmp/{}_{}_{}.png".format(i, dataset_name, binarization_name),
        )


# dataset_name = "mnist"
# shape = (28, 28)

# for binarization_name in ["lt", "mt", "ot"]:
#     save_binary_img(
#         get_ds_example(dataset_name, binarization_name, shape, 3),
#         "{}_{}.png".format(dataset_name, binarization_name),
#     )

