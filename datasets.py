import torch
import torchvision
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import os
import urllib
import errno
from abc import ABC, abstractmethod
import pdb


class SubsetStar(Dataset):
    """
    Subset of a dataset at specified indices.
    Extended version of what is implemented in Pytorch.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __getattr__(self, property):
        # Will be called if the property was not found in the current class
        # Check the original dataset for the specified property
        if hasattr(self.dataset, property):
            attr = getattr(self.dataset, property)
            if "shape" in property.lower():
                attr = len(self), *attr[1:]

            elif "dataframe" in property.lower():
                attr = attr.iloc[self.indices, :]
            return attr
        else:
            raise AttributeError

    def __init__(self, dataset, indices, train, mean=0, std=1):
        self.dataset = dataset
        self.indices = indices
        if train:
            df = self.dataframe
            self.mean = df.mean().values[:-2]
            self.std = df.std().values[:-2]
        else:
            self.mean = mean
            self.std = std

        if not isinstance(self.mean, torch.FloatTensor):
            self.mean = torch.tensor(self.mean, dtype=torch.float32)

        if not isinstance(self.std, torch.FloatTensor):
            self.std = torch.tensor(self.std, dtype=torch.float32)

    def __getitem__(self, idx):
        X, y = self.dataset[self.indices[idx]]
        X = (X - self.mean) / (self.std + 1e-5)
        return X, y

    def __len__(self):
        return len(self.indices)


class SurvivalDataset(ABC, Dataset):
    """
    Abstract class for Survival Datasets
    All datasets should follow the following format :
    $\mathcal{D} = {X_i, lifetime_i, S_i}_{i=1}^N$
    """

    @property
    def shape(self):
        return self.dataframe.shape

    @property
    def xShape(self):
        return (self.dataframe.shape[0], self.dataframe.shape[1] - 2)

    @property
    def yShape(self):
        return (self.dataframe.shape[0], 2)

    def __init__(self, root, window=0, clusterIds=None, download=False):
        self.root = root
        self.window = window

        if download:
            self._download()

        self.dataframe = pd.read_csv(
            os.path.join(self.root, self.fileName), index_col=0
        )

        if self.type == "synthetic":
            if clusterIds is None:
                clusterIds = [1, 2, 4]
            mask = self.dataframe['true_cluster'] < 0
            for i in clusterIds:
                mask |= (self.dataframe['true_cluster'] == i)
            self.dataframe = self.dataframe[mask]

            self.true_cluster = np.array(self.dataframe["true_cluster"])
            self.dataframe = self.dataframe.drop("true_cluster", axis=1)

        # Naming convention: lifetime is called 'x', and timesincelast is called 'dist'.
        columns = list(self.dataframe.columns)
        columns[-1] = "dist"
        columns[-2] = "x"
        self.dataframe.columns = columns

        # Lifetime needs to be an integer
        self.dataframe = self.dataframe.astype({"x": int})

        # t_max (to find the support of the survival distribution). Add 1 for t=0.
        self.tmax = self.dataframe.iloc[:, -2].max() + 1

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        X = torch.tensor(np.array(self.dataframe.iloc[idx, :-2]), dtype=torch.float32)
        lifetime = torch.tensor(
            np.array(self.dataframe.iloc[idx, -2]), dtype=torch.int64
        )
        # s = torch.tensor(np.array(self.dataframe.iloc[idx, -1]), dtype=torch.int64)
        s = torch.tensor(np.array(self.dataframe.iloc[idx, -1]), dtype=torch.float32)
        dead = torch.tensor(np.array((s > self.window) + 0), dtype=torch.int64)

        if self.type == "synthetic":
            true_cluster = torch.tensor(self.true_cluster[idx], dtype=torch.int64)
            sample = (
                X,
                {
                    "lifetime": lifetime,
                    "s": s,
                    "dead": dead,
                    "true_cluster": true_cluster
                },
            )
        else:
            sample = (X, {"lifetime": lifetime, "s": s, "dead": dead})

        return sample

    def _download(self):
        try:
            os.makedirs(self.root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        filePath = os.path.join(self.root, self.fileName)
        try:
            print("Downloading " + self.url + " to " + filePath)
            urllib.request.urlretrieve(self.url, filePath)
        except:
            print("Error downloading the file")


class FriendsterSurvivalDataset(SurvivalDataset):
    """ Friendster Survival Dataset """

    url = 'https://www.dropbox.com/s/jmuvc0b3ls7x2rf/friendsterData.csv?dl=1'
    fileName = "friendsterData.csv"
    units = "months"
    type = "real"


class SyntheticSurvivalDataset(SurvivalDataset):
    url = 'https://www.dropbox.com/s/vs5oghe0izg5z0z/simulationData_124.csv?dl=1'
    fileName = "simulationData_124.csv"
    units = None
    type = "synthetic"


if __name__ == "__main__":

    # Set download to False after downloading the dataset once.
    # data = FriendsterSurvivalDataset('data', window=10, download=True)
    data = SyntheticSurvivalDataset("data", window=0, clusterIds=[1, 2], download=True)

    print(f"Dataset size: {len(data)}")

    # Example on how to use SubsetStar class to create subsets of dataset.
    trainIdx = np.random.choice(len(data), 100, replace=False)
    testIdx = np.random.choice(len(data), 100, replace=False)

    trainData = SubsetStar(data, trainIdx, train=True)
    testData = SubsetStar(
        data, testIdx, train=False, mean=trainData.mean, std=trainData.std
    )

    trainLoader = DataLoader(trainData, batch_size=1028)
    testLoader = DataLoader(testData, batch_size=1028)
