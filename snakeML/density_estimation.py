import numpy
import scipy
from snakeML.numpy_transformations import mrow


def SJoint(ll, pi=None, logarithmic=True):
    """
    Joint density estimation (from loglikelihood)

    Parameters
    ----------
    ll: log-likelihood matrix
    """
    ll = numpy.array(ll)
    classes = ll.shape[0]
    if pi is None:
        pi = numpy.full((classes, 1), 1 / classes)
    pi = numpy.array(pi)
    if logarithmic:
        SJoint = ll + numpy.log(pi.reshape((-1, 1)))
    else:
        SJoint = ll * (pi)
    return SJoint


def SPost(SJoint, SMarginal, logarithmic=True, exp=True):
    """
    Posterior probabilities (Joint density estimation / Marginal density estimation)
    """
    if logarithmic:
        logSPost = SJoint - SMarginal
        if exp:
            return numpy.exp(logSPost)
        else:
            return logSPost
    else:
        return SJoint / SMarginal


def SMarginal(SJoint, logarithmic=True):
    """
    Marginal density estimation (from Joint density estimation)
    """
    if logarithmic:
        return mrow(scipy.special.logsumexp(SJoint, axis=0))
    else:
        return mrow(SJoint.sum(0))


def SPost_from_ll(ll, pi, log=True, return_marginal=False):
    """
    Posterior probabilities from log-likelihood matrix
    returns: posterior probabilities

    Parameters
    ----------
    ll: log-likelihood matrix
    pi: Prior probabilities [Pc1, Pc2, ...]
    """
    SJ = SJoint(ll, pi=pi, logarithmic=log)
    Marginal = SMarginal(SJ, logarithmic=log)
    Posterior = SPost(SJ, Marginal, logarithmic=log)
    if return_marginal:
        return Posterior, Marginal
    else:
        return Posterior


def estimation(ll, log=True, pi=None):
    """
    Estimate class labels from log-likelihoods
    returns: predictions

    Parameters
    ----------
    ll: log-likelihood matrix
    pi: Prior probabilities [Pc1, Pc2, ...]
    """
    SJ = SJoint(ll, pi=pi, logarithmic=log)
    Marginal = SMarginal(SJ, logarithmic=log)
    Posterior = SPost(SJ, Marginal, logarithmic=log)
    pred = numpy.argmax(Posterior, axis=0)
    return pred
