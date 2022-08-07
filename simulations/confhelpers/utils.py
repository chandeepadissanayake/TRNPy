import pandas as pd
from simulations.confhelpers import datasets


def read_dataset(dataset_id: str):
    dataset = pd.read_csv(datasets.DATASET_MAPPINGS[dataset_id], sep="\t", header=None, names=["X1", "X2", "Y"])
    return dataset


def split_dataset(dataset):
    return dataset[["X1", "X2"]], dataset[["Y"]]
