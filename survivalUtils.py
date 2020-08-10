import os
import numpy as np
import torch
from Common import utils
from lifelines.statistics import pairwise_logrank_test, multivariate_logrank_test
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
import logging
from sklearn.metrics import adjusted_rand_score

import matplotlib.pyplot as plt


def findSurvivalDistribution(lifetimes, deads, weights=None):
    """
    Return the survival distribution of a group
    Parameters
    ----------
    lifetimes : Individual lifetimes (can be censored)
    deads : end-of-life signals
    weights : Sample weights (probability of the individual being in this group).

    Returns
    -------
    Survival Distribution (CCDF)

    """

    if type(lifetimes) == np.ndarray and type(deads) == np.ndarray:
        return _findSurvivalDistribution_np(lifetimes, deads, weights)
    elif type(lifetimes) == torch.Tensor and type(deads) == torch.Tensor:
        return _findSurvivalDistribution_torch(lifetimes, deads, weights)
    else:
        raise NotImplementedError


def _findSurvivalDistribution_torch(lifetimes, deads, weights=None):
    """
    Return the survival distribution of a group (for Torch Tensors)
    Parameters
    ----------
    lifetimes : Individual lifetimes (can be censored)
    deads : end-of-life signals
    weights : Sample weights (probability of the individual being in this group).

    Returns
    -------
    Survival Distribution (CCDF)

    """

    if weights is None:
        # If weights not given use, w = 1
        weights = torch.ones_like(lifetimes, dtype=torch.float32)
    freq_lifetimes = utils.bincount(lifetimes, weights)
    freq_lifetimesDead = utils.bincount(lifetimes, weights * deads.float())
    nAlive = utils.reverse(utils.reverse(freq_lifetimes, 0).cumsum(0), 0)

    KMLambda = freq_lifetimesDead / nAlive
    KMProd = (1 - KMLambda).cumprod(0)
    return KMProd


def _findSurvivalDistribution_np(lifetimes, deads, weights=None):
    """
    Return the survival distribution of a group (for Numpy arrays)
    Parameters
    ----------
    lifetimes : Individual lifetimes (can be censored)
    deads : end-of-life signals
    weights : Sample weights (probability of the individual being in this group).

    Returns
    -------
    Survival Distribution (CCDF)

    """

    if weights is None:
        # If weights not given use, w = 1
        weights = np.ones_like(lifetimes, dtype=np.float32)
    freq_lifetimes = np.bincount(lifetimes, weights)
    freq_lifetimesDead = np.bincount(lifetimes, weights * deads)
    nAlive = freq_lifetimes[::-1].cumsum()[::-1]

    KMLambda = freq_lifetimesDead / nAlive
    KMProd = (1 - KMLambda).cumprod(0)
    return KMProd


def _findSurvivalDistrosPerUser(lifetimes, deads, labels):
    """
    Find survival distribution of each cluster and copy it for each user in the cluster.

    Parameters
    ----------
    lifetimes : (Tensor of size N) Individual lifetimes (can be censored)
    deads : end-of-life signals
    labels : Cluster labels

    Returns
    -------
    survivalDistrosPerUser : (Tensor of size N * maxT) with the survival distributions of each user

    """
    maxT = lifetimes.max()
    survivalDistrosPerUser = np.zeros((lifetimes.shape[0], maxT + 1))
    for i in np.unique(labels):
        distro = findSurvivalDistribution(lifetimes[labels == i], deads[labels == i])
        if len(distro) == 0:
            distro = np.pad(
                distro, pad_width=(0, maxT + 1 - distro.shape[0]), mode="constant", constant_values=0.0
            )
        else:
            distro = np.pad(
                distro, pad_width=(0, maxT + 1 - distro.shape[0]), mode="minimum"
            )
        survivalDistrosPerUser[labels == i] = distro

    return survivalDistrosPerUser



def _getTopSurvivalFeatures(
        data, varianceThreshold, xFeature="x", deadFeature="dead"
):
    """
    Get features that are most correlated with survival using CoxPH model.
    Function is used by SemiSupervisedClustering and SupervisedSparseClustering methods.
    """

    _log = logging.getLogger("_log")
    cf = CoxPHFitter(penalizer=1e3)
    nFeatures = data.shape[1]

    # print(np.var(data))
    # Removing features with very low variance that cause convergence problems for Cox Fitter.
    ix = np.unique(list(np.where(np.var(data) > varianceThreshold)[0]) + [data.shape[1]-1, data.shape[1]-2])
    bigvardata = data.iloc[:, ix]


    _log.info('CoxPH fit begins')
    cf.fit(bigvardata, xFeature, event_col=deadFeature)
    _log.info('CoxPH fit ends')

    cf.print_summary()
    hr = np.exp(cf.hazards_)
    sortedFeatures = hr.sort_values(ascending=False)

    # nTop = int(np.ceil(np.sqrt(nFeatures) / 5) * 5)     # Use top sqrt(N) features

    nTop = np.sum(sortedFeatures >= 0.9)        # Use all features with hr>=0.9 (works better than sqrt(N)).

    _log.info(f'Total features in the dataframe: {bigvardata.shape[1]}')
    _log.info(f'Number of features selected: {nTop}')
    cols = sortedFeatures.index[:nTop]
    return cols


def getGoodColumns(df):
    index = []
    currentRank = 0

    for i in range(df.shape[1]):
        rank = np.linalg.matrix_rank(df.iloc[:, index + [i]])
        if "len" in df.columns[i]:
            continue
        if rank > currentRank:
            index.append(i)
            currentRank = rank
        else:
            # print(f"Ignoring {i}th column: {df.columns[i]}")
            # print(df.columns[i])
            pass

    return list(df.columns[index])


def concordanceIndex(lifetimes, deads, labels, nSamplePairs=None):
    """
    Find Concordance Index metric (Harrell's C-index)
    Comments for the implementation taken from Random survival forest paper (C-index calculation)

    Parameters
    ----------
    lifetimes : Individual lifetimes (can be censored)
    deads : end-of-life signals
    labels : Cluster labels
    nSamplePairs : Number of pairs to sample to compute C-index (all pairs might be computationally prohibhitive)

    Returns
    -------
    C-index score

    """

    def isPermissible(lifetimeA, lifetimeB, deadA, deadB):
        if lifetimeA == lifetimeB:
            # Omit pairs i and j if Ti = Tj unless atleast one is a death
            return deadA or deadB
        elif lifetimeA < lifetimeB and not deadA:
            # Omit those pairs whose shorter survival time is censored
            return False
        elif lifetimeB < lifetimeA and not deadB:
            # Omit those pairs whose shorter survival time is censored
            return False
        else:
            return True

    def concordanceValue(
        lifetimeA, lifetimeB, deadA, deadB, expectedLifetimeA, expectedLifetimeB
    ):
        # Assume that the pair is permissible
        if lifetimeA == lifetimeB:
            if deadA and deadB:
                # For each permissible pair, where Ti = Tj and both are deaths,
                # count 1 if predicted outcomes are tied; otherwise, count 0.5
                return 1 if expectedLifetimeA == expectedLifetimeA else 0.5
            else:
                # For each permissible pair, where Ti = Tj but both are not deaths,
                # count 1 if the death has worse predicted outcome; otherwise, count 0.5
                if deadA:
                    return 1 if expectedLifetimeA < expectedLifetimeB else 0.5
                elif deadB:
                    return 1 if expectedLifetimeB < expectedLifetimeA else 0.5
                else:
                    # Should not come here as this pair is not "permissible"
                    return None
        else:
            # For each permissible pair where Ti != Tj,
            # count 1 if the shorter survival time has worse predicted outcome;
            # count 0.5 if predicted outcomes are tied.
            if expectedLifetimeA == expectedLifetimeB:
                return 0.5
            elif lifetimeA < lifetimeB:
                return 1 if expectedLifetimeA < expectedLifetimeB else 0
            elif lifetimeB < lifetimeA:
                return 1 if expectedLifetimeB < expectedLifetimeA else 0
            else:
                # Should not come here
                return None

    if nSamplePairs is None or nSamplePairs <= 0:
        nSamplePairs = 10000

    survivalDistrosPerUser = _findSurvivalDistrosPerUser(lifetimes, deads, labels)
    expectedLifetimes = survivalDistrosPerUser.sum(axis=1)
    permissiblePairs = 0
    concordance = 0
    for i in range(nSamplePairs):
        pairIndex = np.random.choice(lifetimes.shape[0], 2, replace=False)
        A = pairIndex[0]
        B = pairIndex[1]
        if isPermissible(lifetimes[A], lifetimes[B], deads[A], deads[B]):
            permissiblePairs += 1
            concordance += concordanceValue(
                lifetimes[A],
                lifetimes[B],
                deads[A],
                deads[B],
                expectedLifetimes[A],
                expectedLifetimes[B],
            )

    return concordance / permissiblePairs


def brierScore(lifetimes, deads, labels):
    """
    Compute Brier Score
    Parameters
    ----------
    lifetimes : Individual lifetimes (can be censored)
    deads : end-of-life signals
    labels : Cluster labels

    Returns
    -------
    Brier score

    """
    survivalCurves = _findSurvivalDistrosPerUser(lifetimes, deads, labels)
    actualSurvivalCurve = np.zeros_like(survivalCurves)
    for i in range(lifetimes.shape[0]):
        if deads[i] == 1:
            actualSurvivalCurve[i][: (lifetimes[i] + 1)] = 1
        else:
            actualSurvivalCurve[i][:] = 1
    return ((survivalCurves - actualSurvivalCurve) ** 2).mean()


def multivariateLogRankScore(lifetimes, deads, labels):
    """
    Computes LogRank score of all the clusters

    Parameters
    ----------
    lifetimes : Individual lifetimes (can be censored)
    deads : end-of-life signals
    labels : Cluster labels

    Returns
    -------
    LogRank score of all the clusters.

    """

    multivariateLogRankTestResult = multivariate_logrank_test(lifetimes, labels, deads)
    return multivariateLogRankTestResult.test_statistic


def pairwiseLogRankScore(lifetimes, deads, labels):
    """
    Computes Pairwise LogRank score of all the clusters

    Parameters
    ----------
    lifetimes : Individual lifetimes (can be censored)
    deads : end-of-life signals
    labels : Cluster labels

    Returns
    -------
    Pairwise LogRank score of all the clusters in a list

    """

    pairwiseLogrankTestResult = pairwise_logrank_test(lifetimes, labels, deads)
    return pairwiseLogrankTestResult.test_statistic


def concordanceIndexNew(lifetimes, deads, labels):
    """
    Efficient C-Index
    Parameters
    ----------
    lifetimes
    deads
    labels

    Returns
    -------

    """
    survivalDistrosPerUser = _findSurvivalDistrosPerUser(lifetimes, deads, labels)
    expectedLifetimes = survivalDistrosPerUser.sum(axis=1)
    ci = concordance_index(lifetimes, expectedLifetimes, deads)
    return ci


def plotClusterDistributions(
    lifetimes, deads, labels, show=False, save=False, plotFileName=None, units=None
):
    if not show and not save:
        return

    # if save and plotFileName is not None:
        # Save plot information to plot it again later.
        # Save in a file with no extension
        # name, ext = os.path.splitext(plotFileName)
        # torch.save((lifetimes, deads, labels, units), name)

    k = int(max(labels)) + 1
    oneHotLabels = torch.eye(k)[labels].to(lifetimes.device)
    nSamplesPerCluster = oneHotLabels.sum(dim=0).int()
    if show or save:
        maxT = lifetimes.max().item()
        markers = list("o^sd.,v<>12348spP*hH+xXD|_")

        for i in range(k):
            # distro_i = survivalUtils.findSurvivalDistribution(testLifetime, testDead, outs[:, i])
            distro_i = findSurvivalDistribution(lifetimes, deads, oneHotLabels[:, i])
            xPlot = np.insert(np.arange(0, maxT + 1), 0, 0)
            yPlot = np.insert(distro_i.detach().cpu().numpy(), 0, 1.0)
            plt.plot(xPlot, yPlot, alpha=0.5, marker=markers[i], linewidth=2)

        # plt.suptitle(f'Lifetime distribution of clusters (K = {k})', fontsize=24)
        if units is not None:
            plt.xlabel(f"Time (in {units})", fontsize=22)
        else:
            plt.xlabel("Time", fontsize=22)
        plt.ylabel("Probability (CCDF)", fontsize=22)

        # legendList = [f'Cluster {i} (n={nSamplesPerCluster[i-1]})' for i in range(1, k + 1)]
        legendList = [
            r"$\hat{S}_%d$ $(n_%d = %d)$" % (i, i, nSamplesPerCluster[i - 1])
            for i in range(1, k + 1)
        ]

        plt.legend(legendList, fontsize=15, bbox_to_anchor=[1, 1], ncol=1, fancybox=True)

        # plt.legend(legendList, loc='upper center', bbox_to_anchor=(0.5, 1.25),
        #           ncol=3, fancybox=False, shadow=False)

        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.ylim(0, 1.05)
        if save and plotFileName is not None:
            # Save in a file with .pdf extension
            name, ext = os.path.splitext(plotFileName)
            name = name + '.pdf'
            plt.savefig(name, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()


def getAllPlotFileNamesFromBase(basePlotFileName):
    plotInfos = [basePlotFileName + '_train', basePlotFileName + '_test']
    actualPlots = [name + '.pdf' for name in plotInfos]
    return plotInfos + actualPlots



def createLogMetricsCallback(model, trainData, validData, testData, _run):
    """
    Create a callback for logging metrics (only loss).
    :param model: Model
    :param trainData: Training data
    :param validData: Validation data
    :param testData: Test data
    :param _run: Run object to log the metrics to (obtained from Sacred).
    :return: Callback to log the metrics
    """

    def callback_logMetrics(counter, **kwargs):
        """
        Pre-epoch callback for logging experiment metrics
        Parameters
        ----------
        epoch   : Current epoch
        """

        if _run is None:
            return

        if "validLoss" in kwargs:
            validLoss = kwargs["validLoss"]
            _run.log_scalar("valid.loss", validLoss, counter)

        if "trainLoss" in kwargs:
            trainLoss = kwargs["trainLoss"]
            _run.log_scalar("train.loss", trainLoss, counter)

        # testLoss = model.computeLoss(testData, lossOver=model.lossOver)
        # testLoss = float(testLoss.detach().cpu())
        # _run.log_scalar("test.loss", testLoss, epoch)

    return callback_logMetrics


if __name__ == "__main__":
    N = 10000
    print(N)
    k = 3
    lifetimes = np.random.randint(100, size=N)
    dead = np.random.randint(2, size=N)
    labels = np.random.randint(k, size=N)

    w = torch.rand(N, requires_grad=True)

    lrt = pairwiseLogRankScore(lifetimes, dead, labels)

    lifetimes, dead, labels = torch.tensor(lifetimes), torch.tensor(dead), torch.tensor(labels)
    s = findSurvivalDistribution(lifetimes, dead, w)
    print(s)


