import numpy

def oneHotEncoding(labeled_data, return_dictionary= False):
    """
    Returns one-hot encoding of labeled data.
    returns: one-hot encoded data, labels names dictionary

    Parameters
    ----------
    labeled_data: data with labels
    """
    labels={}
    for i, name in enumerate(numpy.unique(labeled_data)):
        labels[name]=i
    new_labels=[]
    for i in labeled_data:
        new_labels.append(labels[i])
    if return_dictionary:
        return numpy.array(new_labels, dtype=numpy.int32), labels
    else:
        return numpy.array(new_labels, dtype=numpy.int32)
    
def znorm(xtrain, xtest=None):
    """
    Returns z-normalized data.

    Parameters
    ----------
    data: data matrix
    """
    mean=xtrain.mean(axis=1).reshape((-1,1))
    deviation=xtrain.std(axis=1).reshape((-1,1))
    if len(xtest)>0:
        return ((xtrain-mean)/deviation), ((xtest-mean)/deviation)
    else:
        return ((xtrain-mean)/deviation)