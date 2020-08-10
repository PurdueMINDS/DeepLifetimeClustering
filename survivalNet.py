import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import signal

import matplotlib.pyplot as plt

import math
import survivalUtils
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from datasets import FriendsterSurvivalDataset, SyntheticSurvivalDataset, SubsetStar
import itertools
from copy import deepcopy

from Common import utils


class SurvivalNet(nn.Module, utils.GracefulKiller):
    """
    Neural Network for Survival Clustering using Empirical Distribution Divergence Maximization

    Given the attributes of the user, learns to soft cluster the users into $K$ groups by maximizing
    the divergence between their empirical survival distributions. Divergence is calculated using a
    differentiable easy-to-compute upper bound of the Kuiper Loss.

    The approach also allows for lack of end-of-life signals.
    """

    def __init__(self, modelParameters):
        super(SurvivalNet, self).__init__()

        layerDims = modelParameters["layerDims"]  # List of layer dimensions
        activationClass = utils.getActivationClassFromString(
            modelParameters["activationClass"]
        )  # Activation class (eg. nn.Tanh, nn.ReLU)
        batchNorm = modelParameters["batchNorm"]  # Use Batch Normalization or not
        self.endOfLifeSignalsLearnt = modelParameters[
            "endOfLifeSignalsLearnt"
        ]  # Indicate presence/lack of end-of-life signals during training
        self.lossName = modelParameters[
            "lossName"
        ]  # Which loss to use : Kuiper variants, KS, MMD

        # Create the appropriate Neural network
        self.net = nn.Sequential()
        for i in range(len(layerDims) - 1):
            inputDim = layerDims[i]
            outputDim = layerDims[i + 1]
            self.net.add_module("fc" + str(i + 1), nn.Linear(inputDim, outputDim))

            if i != len(layerDims) - 2:
                if batchNorm:
                    self.net.add_module("bn" + str(i + 1), nn.BatchNorm1d(outputDim))
                self.net.add_module("activation" + str(i + 1), activationClass())

        self.net.add_module("softmax", nn.Softmax(dim=1))

        # Number of clusters
        self.k = layerDims[-1]

        # Parameter to be optimized ($W_1$ in the paper) to find the probability of end-of-life.
        self.expLambda = torch.nn.Parameter(torch.tensor(1.0))

        # Signals to be handled by the GracefulKiller parent class.
        # At these signals, sets self.kill_now to True
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def forward(self, X):
        outs = self.net(X)
        return outs

    def computeLoss(self, data, nPairs=None, lossName=None):
        """
        Computes the loss

        Parameters
        ----------
        data : SurvivalDataset
            (X : Features
            y : 3 columns with lifetimes, dead and time_till_censoring)
        nPairs: Number of pairs to compute the loss over. Useful for tractability for large K. (default: all pairs)
        lossName : (String) Loss function to use among any of the kuiper variants, KS or MMD. Overrides the lossName specified at initialization.

        Returns
        -------
        Loss value

        """
        if lossName is None:
            lossName = self.lossName

        X, y = data[:]
        lifetime = y["lifetime"]
        s = y["s"]
        dead = y["dead"]

        device = utils.getDevice()

        X, lifetime, s, dead = (
            X.to(device),
            lifetime.to(device),
            s.to(device),
            dead.to(device),
        )

        # End of termination probability
        deadP = 1 - torch.exp(-self.expLambda * s.float())

        outs = self.forward(X)
        distros = {}
        k = outs.shape[1]
        nSamples = torch.sum(outs, dim=0)  # Number of subjects in each cluster

        # Sample a smaller number of clusters to compute loss over instead of (K, 2).
        allPairs = list(itertools.combinations(range(k), 2))

        if nPairs is None or nPairs == "all":
            nPairs = len(allPairs)
        elif nPairs == "k":
            nPairs = k
        else:
            nPairs = int(nPairs)

        nClusterPairsToSample = min(nPairs, len(allPairs))

        clusterPairsToSample = np.random.choice(
            len(allPairs), nClusterPairsToSample, replace=False
        )
        clusterPairsToSample = [allPairs[i] for i in clusterPairsToSample]

        # Compute the lifetime distributions (for only the clusters chosen in this iteration)
        uniqueClusters = np.unique(
            [pair[i] for pair in clusterPairsToSample for i in range(2)]
        )
        for ci in uniqueClusters:
            if not self.endOfLifeSignalsLearnt:
                distros[ci] = survivalUtils.findSurvivalDistribution(
                    lifetime, dead, outs[:, ci]
                )
            else:
                # End-of-life signals are obtained from learnt Exponential CDF
                distros[ci] = survivalUtils.findSurvivalDistribution(
                    lifetime, deadP, outs[:, ci]
                )

        # Using loss as a tensor from beginning to retain the backward graph.
        loss = torch.zeros(len(clusterPairsToSample))
        lossIndex = 0
        for ci, cj in clusterPairsToSample:
            pairLoss = distroLoss(
                distros[ci], nSamples[ci], distros[cj], nSamples[cj], lossName=lossName
            )
            loss[lossIndex] = pairLoss
            lossIndex += 1

        loss = torch.max(loss)
        return loss

    def fit(self, trainData, validData, fitParams, callback_logMetrics):
        """
        Fit the model to the training data
        Parameters
        ----------
        trainData : Train Dataset
        validData : Validation Dataset
        fitParams : Dictionary with parameters for fitting.
                    (lr, weightDecay(l2), lrSchedulerStepSize, fileName, batchSize, lossName, patience, numEpochs)

        Returns
        -------
        None

        """
        logger = utils.getLogger()
        optimizer = optim.Adam(
            self.parameters(), lr=fitParams["lr"], weight_decay=fitParams["weightDecay"]
        )
        fileName = fitParams["fileName"]

        N = trainData.shape[0]
        if fitParams["nMinibatches"] > 0:
            batchSize = 1 << int(math.log2(N // fitParams["nMinibatches"]))
        else:
            batchSize = fitParams["batchSize"]

        if batchSize == 0 or batchSize >= len(trainData):
            batchSize = len(trainData)

        nPairs = fitParams["nPairs"]
        self.nPairs = nPairs

        lossName = self.lossName
        bestValidLoss = 1
        patience = fitParams["patience"]
        numEpochs = fitParams["numEpochs"]
        validationFrequency = 1

        trainLoader = DataLoader(trainData, batch_size=batchSize)

        counter = 1
        for epoch in tqdm(range(1, numEpochs + 1), leave=False, desc="Epochs"):

            # Early-stopping based on loss on a random N pairs.
            validLoss = self.computeLoss(validData, nPairs=nPairs, lossName=lossName)
            validLoss = float(validLoss.detach().cpu())

            if np.isnan(validLoss):
                logger.error("NaNs encountered")
                break

            callback_logMetrics(counter, validLoss=validLoss)

            saved = ""
            if (
                epoch == 1
                or epoch > patience
                or epoch >= numEpochs
                or epoch % validationFrequency == 0
            ) and validLoss < bestValidLoss:
                saved = "(Saved to {})".format(fileName)
                torch.save(self, fileName)
                if validLoss < 0.995 * bestValidLoss:
                    patience = np.max([epoch * 2, patience])
                bestValidLoss = validLoss

            if epoch > patience:
                break

            logger.info(
                "{current} out of {total} : {loss} {saved}".format(
                    current=epoch,
                    total=np.min([patience, numEpochs]),
                    loss=validLoss,
                    saved=saved,
                )
            )

            for batchIdx, batchTrainData in enumerate(
                tqdm(trainLoader, leave=False, desc="Minibatches")
            ):

                optimizer.zero_grad()  # zero the gradient buffer
                batchTrainLoss = self.computeLoss(
                    batchTrainData, nPairs=nPairs, lossName=lossName
                )
                batchTrainLoss.backward()

                callback_logMetrics(
                    counter, trainLoss=float(batchTrainLoss.detach().cpu())
                )
                counter += 1

                nan = 0
                for p in list(filter(lambda p: p.grad is not None, self.parameters())):
                    if torch.isnan(p.grad).any():
                        nan = 1
                        break

                if nan:
                    continue
                optimizer.step()

            if self.kill_now:
                break

    def test(self, testData, metrics=None, show=False, save=False, plotFileName=None):
        """
        Tests the model and plots survival curves

        Parameters
        ----------
        testData : Test Dataset
        metrics : A list of functions that compute a metric given a clustering.
        plotFunctions : Unused
        show : Show the plot
        save : Save the plot

        Returns
        -------
        Return a dictionary with results obtained from each metric

        """

        testX, testy = testData[:]
        testLifetime = testy["lifetime"]
        testDead = testy["dead"]

        device = utils.getDevice()

        testX, testLifetime, testDead = (
            testX.to(device),
            testLifetime.to(device),
            testDead.to(device),
        )

        testProbs = self.forward(testX)

        testLabels = testProbs.max(dim=1)[1]

        results = {}
        if metrics is not None:
            for metric in metrics:
                results[metric.__name__] = metric(
                    testLifetime.cpu().numpy(),
                    testDead.cpu().numpy(),
                    testLabels.detach().cpu().numpy(),
                )

        testLoss = self.computeLoss(
            testData, nPairs=self.nPairs, lossName=self.lossName
        )
        results["loss"] = float(testLoss.detach().cpu())

        if testData.type == "synthetic":
            # The true clusters are available for the synthetic dataset.
            # Compute the Adjusted Rand Index.
            testTrueCluster = testy["true_cluster"]
            results["adjustedRandIndex"] = survivalUtils.adjusted_rand_score(
                testTrueCluster, testLabels.detach().cpu().numpy()
            )

        survivalUtils.plotClusterDistributions(
            testLifetime,
            testDead,
            testLabels,
            show=show,
            save=save,
            plotFileName=plotFileName,
            units=testData.units,
        )
        return results


def distroLoss(distA, nA, distB, nB, lossName="kuiper"):
    """
    Find the distribution loss given two empirical distributions and sample sizes.
    Parameters
    ----------
    distA : (torch.Tensor) Empirical distribution A
    nA : Sample size of A
    distB : (torch.Tensor) Empirical distribution B
    nB : Sample size of B
    lossName : One from (ks, kuiper, kuiper_approx, kuiper_ub, mmd)

    Returns
    -------
    Log distribution loss

    """

    # Assume that distA and distB have the same support (t = 0 to T).
    assert distA.shape[0] == distB.shape[0], "Distributions of different length"
    effectiveN = torch.sqrt((nA * nB) / (nA + nB))

    if lossName == "ks":
        # Find log KS loss between the two KM distributions
        D = torch.abs(torch.max(distA - distB))
        lam = (effectiveN + 0.12 + 0.11 / effectiveN) * D
        lambda_squared = lam ** 2

        if lam < 1e-4:
            # If lambda is too small, return logLoss = 0
            # The sum below would require more terms to converge.
            # As lambda -> 0, we would actually require j -> $\infty$
            logloss = 0

        else:
            kspValue = 0
            for j in range(1, 1000):
                val = (-1) ** (j - 1) * torch.exp(2 * (1 - j * j) * lambda_squared)
                kspValue = kspValue + val

            logKpValue = torch.log(kspValue * 2) - 2 * lambda_squared
            logloss = logKpValue

    if lossName == "mmd":
        gramMatrixSize = distA.shape[0]

        def gaussianKernel(t, t_, sigma=1.0):
            assert (
                len(t.shape) == 2 and t.transpose(0, 1).shape == t_.shape
            ), "Shapes not compatible"
            return torch.exp(-((t - t_) ** 2 / sigma ** 2))

        device = utils.getDevice()

        t = torch.arange(gramMatrixSize).float().to(device)
        K = gaussianKernel(t.unsqueeze(0), t.unsqueeze(1))

        # Find outer products
        aa = torch.ger(distA, distA)
        bb = torch.ger(distB, distB)
        ab = torch.ger(distA, distB)

        mmd = torch.sum(K * (aa + bb - 2 * ab))
        logloss = -torch.log(
            mmd
        )  # We need to maximize the divergence (hence, the negative)

    if lossName.startswith("kuiper"):
        Dplus = torch.max(distA - distB).clamp(min=0)
        Dminus = torch.max(distB - distA).clamp(min=0)
        V = Dplus + Dminus

        logloss = kuiperVariants(effectiveN, V, lossName)

    return logloss


def kuiperVariants(effectiveN, V, lossName="kuiper"):
    lam = (effectiveN + 0.155 + 0.24 / effectiveN) * V
    lambda_squared = lam ** 2

    if lossName == "kuiper_approx":
        # Very basic approximation
        logloss = -lambda_squared

    elif lossName == "kuiper":
        # See numerical recipes book 14.3(?)

        if lam < 1e-4:
            # If lambda is too small, return logLoss = 0
            # The sum below would require more terms to converge.
            # As lambda -> 0, we would actually require j -> $\infty$
            logloss = 0

        else:
            kpValue = 0
            for j in range(1, 1000):
                val = (4 * j * j * lambda_squared - 1) * torch.exp(
                    2 * (1 - j * j) * lambda_squared
                )
                kpValue = kpValue + val

            logKpValue = torch.log(kpValue * 2) - 2 * lambda_squared
            logloss = logKpValue

    elif lossName == "kuiper_ub":
        # See paper for the upper bound calculation
        sqrt2lam = 1 / (math.sqrt(2) * lam)
        r_lower = torch.floor(sqrt2lam)
        r_upper = torch.ceil(sqrt2lam)

        kuiperTerm1 = lambda r, lam_sq: 4 * r ** 2 * lam_sq - 1
        logKuiperTerm2 = lambda r, lam_sq: -2 * r ** 2 * lam_sq

        # Slight difference from the notations in the paper. Here, w and v take lambda_squared as input instead of lambda.
        w = lambda r, lam_sq: -r * torch.exp(logKuiperTerm2(r, lam_sq))
        v = lambda r, lam_sq: kuiperTerm1(r, lam_sq) * torch.exp(
            logKuiperTerm2(r, lam_sq)
        )

        logloss = 0
        if r_lower >= 1:
            logloss = (
                logloss
                + w(r_lower, lambda_squared)
                - w(1, lambda_squared)
                + v(r_lower, lambda_squared)
            )

            logloss = logloss + v(r_upper, lambda_squared) - w(r_upper, lambda_squared)
            logloss = torch.log(logloss)

        else:
            # Here, Lambda is large. The loss simply becomes 0 {because of exp(-2*lambda**2)} and logloss goes to -inf.
            # Hence, can't use this : logloss = logloss + torch.log(v(r_upper, lambda_squared) - w(r_upper, lambda_squared))
            # Use logKuiperTerms to find the logLoss directly.

            logloss = (
                logloss
                + torch.log(kuiperTerm1(r_upper, lambda_squared))
                + logKuiperTerm2(r_upper, lambda_squared)
            )

    return logloss


def boundsForKuiperVariants(N, lossList=None, show=False, save=False):
    """
    Compute and plot the values for various Kuiper variants along with an "almost" exact approximation of Kuiper.

    Parameters
    ----------
    N : Effective sample size of the distributions. Loss is based on a p-value, hence dependent on the sample size.
    lossList : List of losses to compare

    Returns
    -------
    A tensor of size(100, len(lossList)) with values of the losses over a range of values for V (loss statistic).

    """

    if lossList is None:
        lossList = ["kuiper", "kuiper_ub"]  # , 'kuiper_approx']

    listV = torch.arange(0.01, 1, 0.01)
    logKp = torch.zeros(listV.shape[0], len(lossList))

    for j, mode in enumerate(lossList):
        for i, V in enumerate(listV):
            logKp[i][j] = kuiperVariants(N, V, mode)

        plt.plot(listV.numpy(), logKp.numpy()[:, j], label=mode)

    plt.legend()
    if save:
        plt.savefig("plot_kuiperVariants_{N}.pdf".format(N=N), dpi=300)
    if show:
        plt.show()
    plt.close()

    return logKp


def run(trainData, testData, metrics, show, save, _config, _run):
    """
    Run SurvivalNet on given train, test data and compute metrics
    """

    device = utils.getDevice()

    # Use 20% of the training data as validation (for early stopping, hyperparameter tuning)
    trainIdx, validIdx = utils.getRandomSplit(trainData.shape[0], [80, 20])

    validData = SubsetStar(trainData, validIdx, train=False, mean=0.0, std=1.0)

    trainData = SubsetStar(trainData, trainIdx, train=False, mean=0.0, std=1.0)

    # Add Input layer and Output layer to the hidden layers to complete layerDims parameter.
    params = deepcopy(_config)
    params["layerDims"] = [trainData.xShape[1]] + params["layerDims"] + [params["k"]]

    model = SurvivalNet(params).to(device)

    callback_logMetrics = survivalUtils.createLogMetricsCallback(
        model, trainData, validData, testData, _run
    )

    model.train()
    model.fit(
        trainData=trainData,
        validData=validData,
        fitParams=params,
        callback_logMetrics=callback_logMetrics,
    )

    model = torch.load(params["fileName"]).to(device)
    model.eval()

    _, _, plotName_train, plotName_test = survivalUtils.getAllPlotFileNamesFromBase(
        params["plotFileName"]
    )

    trainResults = model.test(
        trainData, metrics=metrics, show=show, save=save, plotFileName=plotName_train
    )

    validResults = model.test(validData, metrics=metrics, show=False, save=False)

    testResults = model.test(
        testData, metrics=metrics, show=show, save=save, plotFileName=plotName_test
    )

    return trainResults, validResults, testResults


if __name__ == "__main__":
    ## USAGE ##
    np.random.seed(42)

    device = utils.getDevice()
    log = utils.getLogger()
    log.info(f"Using device : {device}")

    # data = FriendsterSurvivalDataset("data", window=10, download=False)
    data = SyntheticSurvivalDataset("data", window=0, clusterIds=[1, 2], download=False)

    trainIdx, validIdx, testIdx = utils.getRandomSplit(10000, [70, 10, 20])

    trainData = SubsetStar(data, trainIdx, train=True)
    validData = SubsetStar(
        data, validIdx, train=False, mean=trainData.mean, std=trainData.std
    )
    testData = SubsetStar(
        data, testIdx, train=False, mean=trainData.mean, std=trainData.std
    )

    covariateDim = trainData.xShape[1]

    k = 2
    params = {
        "modelParams": {
            "k": k,
            "layerDims": [covariateDim, 128, 128, k],
            "lossName": "kuiper_ub",
            "activationClass": "tanh",
            "batchNorm": False,
            "endOfLifeSignalsLearnt": False,
        },
        "fitParams": {
            "nPairs": 1,
            "lr": 1e-2,
            "batchSize": 1028,
            "nMinibatches": 8,
            "weightDecay": 0,
            "patience": 100,
            "numEpochs": 10,
            "fileName": f"survivalNet_k={k}.torch",
        },
    }

    metrics = [
        survivalUtils.concordanceIndex,
        survivalUtils.multivariateLogRankScore,
        survivalUtils.brierScore,
    ]


    model = SurvivalNet(params["modelParams"]).to(device)

    callback = survivalUtils.createLogMetricsCallback(model, trainData, validData, testData, None)

    model.train()
    model.fit(trainData=trainData, validData=validData, fitParams=params["fitParams"], callback_logMetrics=callback)
    model = torch.load(params["fitParams"]["fileName"]).to(device)
    model.eval()
    # results = model.test(trainData, metrics=metrics, show=True, save=False)
    # print(results)
    results = model.test(testData, metrics=metrics, show=True, save=False)
    print(results)
