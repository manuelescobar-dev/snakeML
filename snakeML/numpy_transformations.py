import numpy

def mcol(row):
    """
    Returns a column vector from a row vector
    """
    return row.reshape(row.size,1)

def mrow(column):
    """
    Returns a row vector from a column vector
    """
    return column.reshape(1,column.size)

def mean_cov(x, diagCov=False):
    """
    Computes the mean and the covariance matrix of a set of vectors.
    Returns: (mean, covariance matrix)

    Parameters
    ----------
    x: set of vectors
    diagCov: if True, the covariance matrix is a diagonal matrix
    """
    mu=mcol(x.mean(1))
    DC=x-mu
    if diagCov:
         C=numpy.dot(DC,DC.T)/x.shape[1]
         C=C*numpy.identity(C.shape[0])
    else:
        C=numpy.dot(DC,DC.T)/x.shape[1]
    return mu,C

def mean(x):
    """
    Returns the mean of a set of vectors.
    """
    mu=mcol(x.mean(1))
    return mu

def cov(x):
    """
    Returns the covariance matrix of a set of vectors.
    """
    mu=mcol(x.mean(1))
    DC=x-mu
    C=numpy.dot(DC,DC.T)/x.shape[1]
    return C

def wc_cov(x,y):
    """
    Returns the within-class covariance matrix of a set of vectors.
    """
    wcC=numpy.zeros((x.shape[0],x.shape[0]))
    for i in numpy.unique(y):
        D=x[:,y==i]
        mu=mcol(D.mean(1))
        DC=D-mu
        C=numpy.dot(DC,DC.T)
        wcC=numpy.add(wcC,C)
    wcC=wcC/x.shape[1]
    return wcC

