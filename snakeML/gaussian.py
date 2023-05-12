#---------MVG----------
import scipy
import numpy
from snakeML.validation import accuracy, error
import math
from snakeML.density_estimation import logpdf_GAU_ND
from snakeML.numpy_transformations import mean_cov, mrow, wc_cov, mean
import numpy


def SJoint_MVG(Pc, x_train, y_train, x_test, tied=False):
    SJoint=[]
    if tied:
        C=wc_cov(x_train,y_train)
    for i in numpy.unique(y_train):
        if tied:
            mu=mean(x_train[:,y_train==i])
        else:
            mu, C= mean_cov(x_train[:,y_train==i])
        gau= logpdf_GAU_ND(x_test,mu, C, exp=True)
        SJoint.append(gau)
    SJoint=numpy.array(SJoint)
    SJoint=SJoint*(Pc)
    return SJoint

def logSJoint_MVG(Pc, x_train, y_train, x_test, tied=False):
    SJoint=[]
    if tied:
        C=wc_cov(x_train,y_train)
    for i in numpy.unique(y_train):
        if tied:
            mu=mean(x_train[:,y_train==i])
        else:
            mu, C= mean_cov(x_train[:,y_train==i])
        gau= logpdf_GAU_ND(x_test,mu, C)
        SJoint.append(gau)
    SJoint=numpy.array(SJoint)
    SJoint=SJoint+math.log(Pc)
    return SJoint

def SPost_MVG(SJoint, SMarginal):
    P=SJoint/SMarginal
    return P

def logSPost_MVG(logSJoint,logMarginal, exp=True):
    logSPost = logSJoint - logMarginal
    if exp:
        return numpy.exp(logSPost)
    else:
        return logSPost

def SMarginal_MVG(SJoint):
    return mrow(SJoint.sum(0))

def logSMarginal_MVG(logSJoint):
    return  mrow(scipy.special.logsumexp(logSJoint, axis=0))

def MVG(Pc, x_train, y_train, x_test, y_test, tied=False):
    #print("---MVG---")
    SJoint=SJoint_MVG(Pc,x_train,y_train,x_test, tied)
    Marginal=SMarginal_MVG(SJoint)
    Posterior=SPost_MVG(SJoint,Marginal)
    predictions=numpy.argmax(Posterior, axis=0)
    acc=accuracy(predictions,y_test)
    err=error(predictions,y_test)
    return predictions, acc, err

def logMVG(Pc, x_train, y_train, x_test, y_test, tied=False):
    #print("---logMVG---")
    logSJoint=logSJoint_MVG(Pc,x_train,y_train,x_test, tied)
    logMarginal=logSMarginal_MVG(logSJoint)
    p=logSPost_MVG(logSJoint,logMarginal)
    predictions=numpy.argmax(p, axis=0)
    acc=accuracy(predictions,y_test)
    err=error(predictions,y_test)
    return predictions, acc, err

#---------NBG----------

def SJoint_NBG(Pc, x_train, y_train, x_test, tied=False):
    SJoint=[]
    if tied:
        C=wc_cov(x_train,y_train)
        C=C*numpy.identity(C.shape[0])
    for i in numpy.unique(y_train):
        if tied:
            mu=mean(x_train[:,y_train==i])
        else:
            mu, C= mean_cov(x_train[:,y_train==i])
            C=C*numpy.identity(C.shape[0])
        gau= logpdf_GAU_ND(x_test,mu, C, exp=True)
        SJoint.append(gau)
    SJoint=numpy.array(SJoint)
    SJoint=SJoint*(Pc)
    return SJoint

def logSJoint_NBG(Pc, x_train, y_train, x_test, tied=False):
    SJoint=[]
    if tied:
        C=wc_cov(x_train,y_train)
        C=C*numpy.identity(C.shape[0])
    for i in numpy.unique(y_train):
        if tied:
            mu=mean(x_train[:,y_train==i])
        else:
            mu, C= mean_cov(x_train[:,y_train==i])
            C=C*numpy.identity(C.shape[0])
        gau= logpdf_GAU_ND(x_test,mu, C)
        SJoint.append(gau)
    SJoint=numpy.array(SJoint)
    SJoint=SJoint+math.log(Pc)
    return SJoint

def NBG(Pc, x_train, y_train, x_test, y_test, tied=False):
    #print("---NBG---")
    SJoint=SJoint_NBG(Pc,x_train,y_train,x_test, tied)
    Marginal=SMarginal_MVG(SJoint)
    Posterior=SPost_MVG(SJoint, Marginal)
    predictions=numpy.argmax(Posterior, axis=0)
    acc=accuracy(predictions,y_test)
    err=error(predictions,y_test)
    return predictions, acc, err

def logNBG(Pc, x_train, y_train, x_test, y_test, tied=False):
    #print("---logNBG---")
    logSJoint=logSJoint_NBG(Pc,x_train,y_train,x_test, tied)
    logMarginal=logSMarginal_MVG(logSJoint)
    p=logSPost_MVG(logSJoint,logMarginal)
    predictions=numpy.argmax(p, axis=0)
    acc=accuracy(predictions,y_test)
    err=error(predictions,y_test)
    return predictions, acc, err

def generativeClassifier(x_train, y_train, x_test, y_test, model, Pc=False):
    classes=numpy.unique(y_train)
    if not Pc:
        Pc=1/classes.size
    match(model):
        case("MVG"):
            return logMVG(Pc,x_train,y_train,x_test,y_test)
        case("logMVG"):
            return logMVG(Pc,x_train,y_train,x_test,y_test)
        case("NBG"):
            return NBG(Pc,x_train,y_train,x_test,y_test)
        case("logNBG"):
            return logNBG(Pc,x_train,y_train,x_test,y_test)
        case("TiedMVG"):
            return MVG(Pc,x_train,y_train,x_test,y_test, tied=True)
        case("logTiedMVG"):
            return logMVG(Pc,x_train,y_train,x_test,y_test, tied=True)
        case("TiedNBG"):
            return NBG(Pc,x_train,y_train,x_test,y_test, tied=True)
        case("logTiedNBG"):
            return logNBG(Pc,x_train,y_train,x_test,y_test, tied=True)

def kfold_generativeClassifier(D, L, model):
    error=0
    indexes = numpy.arange(D.shape[1])
    for i in range(D.shape[1]):
        x_train = D[:, indexes!=i]
        y_train = L[indexes!=i]
        x_test = D[:, indexes==i]
        y_test = L[indexes==i]
        pred, acc, err=generativeClassifier(x_train,y_train,x_test,y_test, model)
        error+=err
    return(error/D.shape[1])