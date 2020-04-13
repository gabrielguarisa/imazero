import numpy as np
import pickle as pk
import wisardpkg as wp
import os
from .preprocessing import get_preprocessing, LabelEncoder


class Dataset:
    def __init__(
        self,
        train,
        test,
        dataset_name,
        binarization_name,
        num_classes=0,
        entry_size=0,
        shape=None,
    ):
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

        self.train = train
        self.test = test

        for i in range(len(self.train)):
            self.X_train.append(self.train[i].list())
            self.y_train.append(self.train.getLabel(i))

        for i in range(len(self.test)):
            self.X_test.append(self.test[i].list())
            self.y_test.append(self.test.getLabel(i))

        self.X_train = np.array(self.X_train, dtype=np.uint8)
        self.y_train = np.array(self.y_train, dtype=np.uint8)
        self.X_test = np.array(self.X_test, dtype=np.uint8)
        self.y_test = np.array(self.y_test, dtype=np.uint8)

        self.dataset_name = dataset_name
        self.binarization_name = binarization_name

        if num_classes == 0:
            self.num_classes = len(np.unique(train.getLabels()))
        else:
            self.num_classes = num_classes

        if entry_size == 0:
            self.entry_size = len(self.train[0])
        else:
            self.entry_size = entry_size

        self.shape = shape

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_name(self):
        return "{}_{}".format(self.dataset_name, self.binarization_name)

    def __str__(self):
        return "{}_{}".format(self.dataset_name, self.binarization_name)

    def save(self, folder):
        if folder[-1] != "/":
            folder = "{}/".format(folder)
        folder = "{}{}/".format(folder, self.dataset_name)
        os.makedirs(folder, exist_ok=True)

        train_filename = "{}_train".format(self.binarization_name)
        test_filename = "{}_test".format(self.binarization_name)
        info_filename = "{}_info.ds".format(self.binarization_name)

        self.train.save("{}{}".format(folder, train_filename))
        self.test.save("{}{}".format(folder, test_filename))

        with open("{}{}".format(folder, info_filename), "wb") as output_file:
            pk.dump(
                {
                    "train_filename": "{}.wpkds".format(train_filename),
                    "test_filename": "{}.wpkds".format(test_filename),
                    "num_classes": self.num_classes,
                    "entry_size": self.entry_size,
                    "shape": self.shape,
                },
                output_file,
            )
        return self

    @staticmethod
    def load(dataset_name, binarization_name, folder):
        if folder[-1] != "/":
            folder = "{}/".format(folder)
        folder = "{}{}/".format(folder, dataset_name)

        filename = "{}{}".format(folder, binarization_name)

        data = {}
        with open("{}_info.ds".format(filename), "rb") as input_file:
            data = pk.load(input_file)
        return Dataset(
            train=wp.DataSet("{}{}".format(folder, data["train_filename"])),
            test=wp.DataSet("{}{}".format(folder, data["test_filename"])),
            dataset_name=dataset_name,
            binarization_name=binarization_name,
            num_classes=data["num_classes"],
            entry_size=data["entry_size"],
            shape=data["shape"],
        )

    @staticmethod
    def binarize(
        train_images,
        train_labels,
        test_images,
        test_labels,
        dataset_name,
        binarization_name,
        num_classes=0,
        entry_size=0,
    ):
        shape = None
        if np.array(train_images[0]).ndim == 2:
            shape = np.shape(train_images[0])
        method = get_preprocessing(binarization_name)

        binary_train_images = method.transform(train_images)
        binary_test_images = method.transform(test_images)

        if type(train_labels[0]) == str:
            le = LabelEncoder()
            le.fit(train_labels)
            binary_train_labels = le.transform(train_labels)
            binary_test_labels = le.transform(test_labels)
        else:
            binary_train_labels = train_labels
            binary_test_labels = test_labels

        return Dataset(
            train=wp.DataSet(
                binary_train_images, np.array(binary_train_labels).astype(str)
            ),
            test=wp.DataSet(
                binary_test_images, np.array(binary_test_labels).astype(str)
            ),
            dataset_name=dataset_name,
            binarization_name=binarization_name,
            num_classes=num_classes,
            entry_size=entry_size,
            shape=shape,
        )
