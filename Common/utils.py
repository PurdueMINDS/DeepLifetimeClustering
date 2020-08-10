import numpy as np
import torch
import os
import errno
import logging
from tqdm.auto import tqdm
from pprint import pformat
import tempfile
import random


################################ Torch helpers ########################################
def getDevice():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def getLogger():
    log = logging.getLogger("_log")

    if log.handlers == []:
        tqdmHandler = TqdmLoggingHandler()  # Works with tqdm progress bars.
        tqdmHandler.setFormatter(logging.Formatter("[%(levelname)s]: %(message)s"))
        log.addHandler(tqdmHandler)
        log.setLevel("INFO")

    return log


def reverse(x, dim):
    """
    Reverses a tensor along the given dimension
    :param x: Torch tensor
    :param dim: Dimension along which to reverse
    :return: Reversed tensor
    """
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def makedirs(root):
    """
    Create directory (recursively) if it does not exist
    :param root: String
    :return: None
    """
    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise




class TqdmLoggingHandler(logging.Handler):
    """
    A logging handler to allow printing in between steps of tqdm bar.
    """
    def __init__(self, level=logging.NOTSET):
        super(self.__class__, self).__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


class GracefulKiller:
    """
    Handles signals (like KeyboardInterrupt) gracefully
    Usage: Inherit from this class, add signal.signal() function with self.exit_gracefully as the handler.
           Sets self.kill_now to True if signal received.
    """
    kill_now = False
    def __init__(self):
        pass

    def exit_gracefully(self, signum, frame):
        logger = logging.getLogger("_log")
        logger.setLevel(logging.DEBUG)
        if not self.kill_now:
            logger.info("Stop signal received. Killing now...")
        else:
            logger.info("Processing. Please wait.")
        self.kill_now = True


def logInfoDict(logger, dict, text):
    """
    Logs (at level INFO) a dictionary in a pretty way (over different lines while logging).
    :param logger: Logger object
    :param dict: Dictionary to print
    :param text: Title
    :return:
    """
    logger.info("=" * 80)
    logger.info(text)
    for line in pformat(dict).split("\n"):
        logger.info(line)
    logger.info("=" * 80)


def flatten_dict(d):
    """
    Flattens a dictionary. Keys within keys are concatenated using '.'.
    :param d: Dictionary to flatten
    :return: Flattened dictionary
    """
    def items():
        for key, value in d.items():
            if isinstance(value, dict):
                for subkey, subvalue in flatten_dict(value).items():
                    yield key + "." + subkey, subvalue
            else:
                yield key, value

    return dict(items())


def removeKeysRecursive(value, removeKeys):
    if type(value) != dict:
        return value

    newValue = dict()
    for k, v in value.items():
        if k not in removeKeys:
            newValue[k] = removeKeysRecursive(v, removeKeys)

    return newValue


def bincount(x, w=None):
    """
    Bincount for torch (allowing for gradients)
    :param x: A torch tensor with integers
    :param w: Optional weights
    :return: A tensor (result) of size max(x) + 1 with (weighted) counts of elements in x

    Example Usage:
    # N = 100000
    # x = torch.randint(100, (N, )).int()
    # b = bincount(x)
    """

    if w is None:
        w = torch.ones_like(x, dtype=torch.float32)
    m = x.max().item() + 1

    bincount = torch.zeros(m, dtype=torch.float32, device=x.device)

    for i, xi in enumerate(x):
        bincount[xi] = bincount[xi] + w[i]

    return bincount


def createModelAndPlotFiles(folder):
    """
    Create temporary files for model and plot
    :param folder: Folder to create temporary files in
    :return:
    """

    _log = logging.getLogger("_log")
    _log.warning("Function deprecated. Use createTempFiles.")

    makedirs(folder)
    tempFileHandle, tempFileName = tempfile.mkstemp(dir=folder)
    tempPlotFileHandle, tempPlotFileName = tempfile.mkstemp(suffix='.pdf', dir=folder)

    return tempFileName, tempPlotFileName


def createTempFiles(folder, suffixList=[]):
    """
    Create temporary files for model and plot
    :param folder: Folder to create temporary files in
    :param suffixList: List of suffixes. Model files have no suffix, plot files need to have '.pdf' suffix.

    """

    makedirs(folder)
    tempFiles = []

    for suffix in suffixList:
        _, tempFileName = tempfile.mkstemp(suffix=suffix, dir=folder)
        tempFiles.append(tempFileName)

    return tempFiles



def getRandomSplit(N, splitPercentages):
    """
    Get random indices to split the data according to the percentages specified
    :param N: Generate a permutation from (0, N-1)
    :param splitPercentages: Percentages to split
    :return: List of indices with lengths according to splitPercentages
    """
    assert np.sum(splitPercentages) == 100, 'Percentages should sum to 100'

    indices = np.random.permutation(N)
    splitValues = (np.cumsum(splitPercentages[:-1]) * N / 100).astype(int)
    return np.split(indices, splitValues)


def getActivationClassFromString(activationClassString):
    """

    Parameters
    ----------
    activationClassString: String like "ReLU, Tanh"

    Returns
    -------
    Activation class

    """
    import torch.nn as nn
    if activationClassString is None:
        return None

    activationDictionary = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "softmax": nn.Softmax,
        "none": None
    }

    return activationDictionary[activationClassString.lower()]


def random_product(*args, **kwargs):
    """Draw an item at random from each of the input iterables.
    This equivalent to taking a random selection from
    ``itertools.product(*args, **kwargs)``.
    """

    pools = [tuple(pool) for pool in args] * kwargs.get('repeat', 1)
    while True:
        yield tuple(random.choice(pool) for pool in pools)


def detachAndFloat(a):
    return float(a.detach().cpu())

def setInterrupted(_run):
    _run.meta_info["my_status"] = "INTERRUPTED"
