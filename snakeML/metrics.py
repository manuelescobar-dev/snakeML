import numpy
import matplotlib.pyplot as plt

def matrix_max_error(m1,m2):
    """
    Used to compare two matrices, one of them is the expected result and the other is the result of a function

    Parameters
    ----------
    m1: expected result
    m2: result of a function
    """
    print("Error: ",numpy.abs(m1 - m2).max())

def accuracy(predicted, test):
    """
    Computes the accuracy of the predicted values
    returns: accuracy (%)

    Parameters
    ----------
    predicted: predicted values
    test: expected values
    """
    correct=0
    for i in range(len(predicted)):
        if predicted[i]==test[i]:
            correct+=1
    return correct/len(predicted)*100

def error(predicted, test):
    """
    Computes the error of the predicted values
    returns: error (%)

    Parameters
    ----------
    predicted: predicted values
    test: expected values
    """
    wrong=0
    for i in range(len(predicted)):
        if predicted[i]!=test[i]:
            wrong+=1
    #print("Error: ",(wrong/len(predicted))*100, "%")
    return wrong/len(predicted)*100

def confusion_matrix(predicted, y_test, k=None):
    """
    Computes the confusion matrix of the predicted values
    returns: confusion matrix (kxk list)
    
    Parameters
    ----------
    predicted: predicted values
    test: expected values
    k: number of classes
    """
    k1=numpy.max(numpy.unique(y_test))
    k2=numpy.max(numpy.unique(predicted))
    k=max(int(k1),int(k2))
    if k<1:
        k=1
    cm=numpy.zeros((k+1,k+1))
    for i in range(len(predicted)):
        cm[int(predicted[i]),int(y_test[i])]=cm[int(predicted[i]),int(y_test[i])]+1
    return cm

def optimalBayesDecision(ll, pi=None, C=None, t=None):
    """
    Computes the solution of the optimal Bayes decision rule, given the log-likelihood ratio, the prior probabilities and the cost matrix
    returns: predicted values

    Parameters
    ----------
    ll: log-likelihood ratio
    pi: prior probabilities
    C: cost matrix
    t: threshold (optional)
    """
    ll=numpy.array(ll)
    if ll.ndim>1:
        if ll.shape[1]>1 and ll.shape[0]>1:
            ll= ll[1,:]-ll[0,:]
    pred=numpy.zeros(ll.size)
    if t is None:
        if not pi or not C:
            raise Exception("Enter pi and C or t")
        else:
            t=-numpy.log((pi[1]*C[0][1])/(pi[0]*C[1][0]))
    pred[ll.flatten()>(t)]=1
    return pred

def DCF( pi=None, C=None, conf_matrix=None, pred=None, ytest=None, normalized=True, wps=None):
    """
    Computes the Detection Cost Function (DCF) given the prior probabilities and the cost matrix
    returns: DCF

    Parameters
    ----------
    pi: prior probabilities
    C: cost matrix
    conf_matrix: confusion matrix (optional)
    pred: predicted values (optional)
    ytest: expected values (optional)
    normalized: normalized DCF (optional)
    """
    if wps is None:
        if conf_matrix is None:
            if len(pred)>0 and len(ytest)>0:
                conf_matrix=confusion_matrix(pred, ytest)
            else:
                raise Exception("Enter conf_matrix or pred and ytest")
        if len(C)==2:
            if (conf_matrix[0][1]+conf_matrix[1][1])==0:
                FNR=0
            else:  
                FNR=conf_matrix[0][1]/(conf_matrix[0][1]+conf_matrix[1][1])
            if (conf_matrix[0][0]+conf_matrix[1][0])==0:
                FPR=0
            else:
                FPR=conf_matrix[1][0]/(conf_matrix[0][0]+conf_matrix[1][0])
            dcf=pi[1]*C[0][1]*FNR+pi[0]*C[1][0]*FPR
            if normalized:
                return dcf/min(pi[1]*C[0][1],pi[0]*C[1][0])
            else:
                return dcf
        cms=numpy.array(conf_matrix).sum(axis=0)
        r=conf_matrix/cms
        rc=numpy.multiply(r.T, C)
        rc_s=rc.sum(axis=1)
        rcs=numpy.multiply(rc_s,pi).sum(axis=0)
        if normalized:
            cp=numpy.min(numpy.dot(C, pi))
            return rcs/cp
        else:
            return rcs
    else:
        dcfs=[]
        for i in wps:
            pi=i[0]
            C=i[1]
            if conf_matrix is None:
                if len(pred)>0 and len(ytest)>0:
                    conf_matrix=confusion_matrix(pred, ytest)
                else:
                    raise Exception("Enter conf_matrix or pred and ytest")
            if (conf_matrix[0][1]+conf_matrix[1][1])==0:
                FNR=0
            else:  
                FNR=conf_matrix[0][1]/(conf_matrix[0][1]+conf_matrix[1][1])
            if (conf_matrix[0][0]+conf_matrix[1][0])==0:
                FPR=0
            else:
                FPR=conf_matrix[1][0]/(conf_matrix[0][0]+conf_matrix[1][0])
            dcf=pi[1]*C[0][1]*FNR+pi[0]*C[1][0]*FPR
            if normalized:
                ndcf=dcf/min(pi[1]*C[0][1],pi[0]*C[1][0])
                dcfs.append(ndcf)
            else:
                dcfs.append(dcf)
        return dcfs
    
def minDCF(llr, wp, y_test, return_effective_priors=False, normalized=True):
    """
    Computes the minimum DCF given the log-likelihood ratio, the working point and the expected values.
    returns: (numpy array with minDCF for each working point, effective priors)
    
    Parameters
    ----------
    llr: log-likelihood ratio
    wp: working point [(wp1_priors, wp1_cost_matrix), ...]
        Example: wp=[([0.5,0.5],[[0,1],[1,0]]),([0.1,0.9],[[0,1],[1,0]]),([0.9,0.1],[[0,1],[1,0]])]
    y_test: expected values
    """
    ll=numpy.array(llr)
    if ll.ndim>1:
        if ll.shape[1]>1 and ll.shape[0]>1:
            ll= ll[1,:]-ll[0,:]
    thresholds=[-numpy.inf, numpy.inf]
    thresholds.extend(ll.flatten())
    thresholds=numpy.sort(thresholds)
    dcfs=numpy.zeros((len(wp),len(thresholds)))
    for i in range(len(thresholds)):
        for j in range(len(wp)):
            pred=optimalBayesDecision(ll, t=thresholds[i])
            dcf=DCF(wp[j][0], wp[j][1], pred=pred, ytest=y_test,normalized=normalized)
            dcfs[j][i]=dcf
    if return_effective_priors:
        return numpy.min(dcfs, axis=1), numpy.argmin(dcfs, axis=1)
    else:
        return numpy.min(dcfs, axis=1)

def binary_DCF_minDCF(ll, pi, C, y_test):
    """
    Computes the DCF and the minDCF given the log-likelihood ratio, the prior probabilities, the cost matrix and the expected values.
    returns: (DCF, minDCF)

    Parameters
    ----------
    ll: log-likelihood ratio
    pi: prior probabilities
    C: cost matrix
    y_test: expected values
    """
    pred=optimalBayesDecision(ll, pi, C)
    cm=confusion_matrix(pred, y_test)
    dcf=DCF(cm, pi, C)
    mindcf=minDCF(ll, pi, C, y_test)
    return dcf, mindcf

def ROC(ll, y_test, submodels):
    """
    Plots the ROC curve given the log-likelihood ratio and the expected values.

    Parameters
    ----------
    ll: log-likelihood ratio
    y_test: expected values
    """
    plt.figure()
    if type(ll) is not list:
        thresholds=[-numpy.inf, numpy.inf]
        thresholds.extend(ll[i])
        thresholds=numpy.sort(thresholds)
        x=[]
        y=[]
        for i in thresholds:
            pred=optimalBayesDecision(ll[i], t=i)
            cm=confusion_matrix(pred, y_test)
            x.append(cm[1][0])
            y.append(cm[1][1])
        #plot ROC curve
        plt.plot(x/max(x),y/max(y))
    else:
        for j in range(len(ll)):
            thresholds=[-numpy.inf, numpy.inf]
            if type(ll[j]) is not list:
                if ll[j].ndim>1:
                    if ll[j].shape[1]>1 and ll[j].shape[0]>1:
                        ll[j]= ll[j][1,:]-ll[j][0,:]
            thresholds.extend(ll[j])
            thresholds=numpy.sort(thresholds)
            x=[]
            y=[]
            for i in thresholds:
                pred=optimalBayesDecision(ll[j], t=i)
                cm=confusion_matrix(pred, y_test)
                x.append(cm[1][0])
                y.append(cm[1][1])
            #plot ROC curve
            plt.plot(x/max(x),y/max(y), label=submodels[j])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()

def bayes_error_plot(scores, y_test, submodels, c=[[0,1],[1,0]], precision=0.25, effPriorLB=-5, effPriorUB=5, save_image=True, filename="bayes_error_plot"):
    """
    Plots the Bayes error given the log-likelihood ratio, the expected values, the cost matrix and the precision of the threshold.

    Parameters
    ----------
    scores: log-likelihood ratio
    y_test: expected values
    c: cost matrix
    precision: precision of the threshold
    effPriorLB: lower bound of the logarithmic effective prior
    effPriorUB: upper bound of the logarithmic effective prior
    save_image: save image (True) or show image (False)
    filename: name of the image
    """
    effPriorLogOdds=numpy.arange(effPriorLB,effPriorUB+precision,step=precision)
    colors=["blue","green","red","purple","brown","pink","gray","olive","cyan"]
    plt.figure()
    for i in range(len(scores)):
        dcf=[]
        mindcf=[]
        effPriors=[]
        for p in effPriorLogOdds:
            s=scores[i]
            eff=numpy.exp(p)/(1+numpy.exp(p))
            effPi=[1-eff, eff]
            effPriors.append(effPi)
            pred=optimalBayesDecision(s, effPi, c)
            cm=confusion_matrix(pred, y_test)
            dcf.append(DCF(effPi, c, conf_matrix=cm))
            mindcf.append(minDCF(s,[(effPi,c)],y_test))
        print("plotting...")
        plt.plot(effPriorLogOdds, dcf, label=submodels[i]+" (act. DCF)", color=colors[i])
        plt.plot(effPriorLogOdds, mindcf, label=submodels[i]+" (min. DCF)",linestyle="dashed", color=colors[i])
    plt.legend()
    plt.ylim([0, 1])
    plt.xlim([effPriorLB, effPriorUB])
    plt.xlabel("threshold")
    plt.ylabel("minDCF")
    #print("effPrior of min:",effPriors[mindcf.index(min(mindcf))])
    if save_image:
        plt.savefig(filename+".png")
    else:
        plt.show()

def test_metrics(predicted=None, y_test=None, metrics=["Accuracy"], args=None):
    """
    Computes the metrics given the predicted values, the expected values and the metrics to be computed.
    returns: list of metrics

    Parameters
    ----------
    predicted: predicted values
    y_test: expected values
    metrics: list of metrics to be computed
        - Accuracy
        - Error
        - DCF
        - minDCF
        - minAndAvgDCF
        - avgMinDCF
    """
    testm=[]
    for i in metrics:
        if type(i) is tuple:
            metric=i[0]
        else:
            metric=i
        m=0
        if metric=="Accuracy":
            m=accuracy(predicted, y_test)
        elif metric=="Error":
            m=error(predicted, y_test)
        elif metric=="DCF":
            wp=i[1]
            if wp:
                if predicted is None or y_test is None:
                    m=DCF(wps=wp)
                else:
                    m=DCF(wps=wp, pred=predicted, ytest=y_test)
            else:
                raise Exception("Missing working point")
        elif metric=="avgDCF":
            wp=i[1]
            if wp:
                if predicted is None or y_test is None:
                    m=DCF(wps=wp)
                    m=numpy.mean(m)
                else:
                    m=DCF(wps=wp, pred=predicted, ytest=y_test)
                    m=numpy.mean(m) 
            else:
                raise Exception("Missing working point")
        elif metric=="minDCF":
            wp=i[1]
            ll=args
            if len(wp)>0 and len(ll)>0:
                m=minDCF(ll, wp, y_test=y_test)
            else:
                raise Exception("Missing working point or ll")
        elif metric=="minAndAvgDCF":
            wp=i[1]
            ll=args
            if len(wp)>0 and len(ll)>0:
                m=minDCF(ll, wp, y_test=y_test)
                m=numpy.append(m, numpy.mean(m))
            else:
                raise Exception("Missing working point or ll")
        elif metric=="avgMinDCF":
            wp=i[1]
            ll=args
            if len(wp)>0 and len(ll)>0:
                m=numpy.mean(minDCF(ll, wp, y_test=y_test))
            else:
                raise Exception("Missing working point or ll")
        testm.append(m)
    return testm