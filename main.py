import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import (
    FriendsterSurvivalDataset,
    SyntheticSurvivalDataset,
    SubsetStar,
)
import random
from Common import utils
import sys
import logging
import os
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
import survivalUtils
import survivalNet

import argparse


def crossValidationSplit(lenData, reset=False):
    """
    Setup cross-validation folds

    Parameters
    ----------
    lenData: Length of the dataset
    Rest of the parameters come from the configuration

    """

    # Filename to save the cross-validation folds
    cvFilename = os.path.join(
        "data", f"cv_{args.dataset}_nFolds{args.cvFolds}_seed{args.seed}.bin"
    )

    if not reset and os.path.isfile(cvFilename):
        # Use the file if exists
        log.info(f"Using CV splits file : {cvFilename}")
        foldsIndices = torch.load(cvFilename)
    else:
        # Create the file with cross-validation folds
        log.info(f"CV splits file {cvFilename} not found. Creating one.")
        permutation = torch.randperm(lenData)
        foldSize = lenData // args.cvFolds
        foldsIndices = []
        for iFold in range(args.cvFolds):
            start = iFold * foldSize
            end = start + foldSize
            foldsIndices.append(list(permutation[start:end]))
            torch.save(foldsIndices, cvFilename)

    # Create train/test splits based on the fold and cvIteration
    trainIdx = []
    testIdx = []
    for i, fold in enumerate(foldsIndices):
        if i == args.cvIt:
            testIdx.extend(fold)
        else:
            trainIdx.extend(fold)

    return trainIdx, testIdx



# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log = utils.getLogger()


# Training settings
parser = argparse.ArgumentParser(description='Deep Lifetime Clustering.')
parser.add_argument('--dataset', type=str, default=None, metavar='N',
                    help='Dataset (friendster or synthetic)')
parser.add_argument('--download', action='store_true', default=False,
                    help='Download the dataset')
parser.add_argument('--k', type=int, default=2, metavar='N',
                    help='Number of clusters (default: 2)')

parser.add_argument('--lossName', type=str, default="kuiper_ub", metavar='N',
                    help='Loss for Lifetime clustering: kuiper_ub or mmd')
parser.add_argument('--eol', action='store_true', default=False,
                    help='End of life signals learnt')


parser.add_argument('--cvFolds', type=int, default=5, metavar='N',
                    help='Number of cross-validation folds (default: 5)')
parser.add_argument('--cvIt', type=int, default=0, metavar='N',
                    help='Run fold i (default: 0)')

parser.add_argument('--Ntrain', type=int, default=10000, metavar='N',
                    help='Number of training samples from each CV-fold (default: 10000)')
parser.add_argument('--Ntest', type=int, default=-1, metavar='N',
                    help='Number of test samples from each CV-fold, only used for debugging faster (default: -1)')
parser.add_argument('--batchSize', type=int, default=1028, metavar='N',
                    help='input batch size for training (default: 1028)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')

parser.add_argument('--show', action='store_true', default=False,
                       help='Show results plots')
parser.add_argument('--save', action='store_true', default=False,
                    help='Save results plots')


args = parser.parse_args()



# Set seed for random, numpy and torch (for reproducibility)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

log.info(f"Using {device}")


if args.dataset.lower().startswith("friendster"):
    data = FriendsterSurvivalDataset("data", window=10, download=args.download)
elif args.dataset.lower().startswith("synthetic"):
    data = SyntheticSurvivalDataset("data", window=0, clusterIds=[1, 2], download=args.download)
else:
    raise NotImplementedError


params = {
    # Model params
    "k": args.k,
    "layerDims": [128, 128],
    "lossName": args.lossName,
    "activationClass": "relu",
    "batchNorm": False,
    "endOfLifeSignalsLearnt": args.eol,
    "nPairs": "k",

    # Fit params
    "lr": args.lr,
    "nMinibatches": -1,
    "batchSize": args.batchSize,
    "weightDecay": 0,
    "patience": max(10, args.epochs// 10),  # Patience for early stopping
    "numEpochs": args.epochs,
    "fileName": f"survivalNet_k={args.k}.torch",
    "plotFileName": "plot",
}


utils.logInfoDict(log, params, "Configuration: ")

# Cross validation split according to the configuration parameters.
trainIdx, testIdx = crossValidationSplit(len(data), reset=False)
trainIdx = np.random.choice(
    trainIdx, len(trainIdx) if args.Ntrain == -1 else args.Ntrain, replace=False
)
testIdx = np.random.choice(
    testIdx, len(testIdx) if args.Ntest == -1 else args.Ntest, replace=False
)

trainData = SubsetStar(data, trainIdx, train=True)
testData = SubsetStar(
    data, testIdx, train=False, mean=trainData.mean, std=trainData.std
)

# Metrics to evaluate on
metrics = [
    survivalUtils.concordanceIndex,
    survivalUtils.multivariateLogRankScore,
    survivalUtils.brierScore,
]

params["fileName"] = f"survivalNet_k={args.k}.torch"

# Run the model
log.info("Begin Run : SurvivalNet")
runFunction = survivalNet.run
trainResults, validResults, testResults = runFunction(
    trainData, testData, metrics, show=args.show, save=args.save, _config=params, _run=None
)

# Print train and test results
utils.logInfoDict(log, trainResults, "TrainResults: ")
utils.logInfoDict(log, testResults, "TestResults: ")

