import numpy

def mcol(row):
    return row.reshape(row.size,1)

def mrow(column):
    return column.reshape(1,column.size)

def mean_cov(x, diagCov=False):
    mu=mcol(x.mean(1))
    DC=x-mu
    if diagCov:
         C=numpy.dot(DC,DC.T)/x.shape[1]
         C=C*numpy.identity(C.shape[0])
    else:
        C=numpy.dot(DC,DC.T)/x.shape[1]
    return mu,C

def mean(x):
    mu=mcol(x.mean(1))
    return mu

def cov(x):
    mu=mcol(x.mean(1))
    DC=x-mu
    C=numpy.dot(DC,DC.T)/x.shape[1]
    return C

def wc_cov(x,y):
    wcC=numpy.zeros((x.shape[0],x.shape[0]))
    for i in numpy.unique(y):
        D=x[:,y==i]
        mu=mcol(D.mean(1))
        DC=D-mu
        C=numpy.dot(DC,DC.T)
        wcC=numpy.add(wcC,C)
    wcC=wcC/x.shape[1]
    return wcC

