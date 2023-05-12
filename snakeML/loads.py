import sklearn.datasets
from snakeML.numpy_transformations import mcol, mrow
import numpy
from snakeML.preprocessing import oneHotEncoding

def loadData(filename, row_attributes= False, labels= False, separator=',', numpyDataType=None):
    data=[]
    with open(filename) as f:
        if labels:
            labels=[]
            for line in f:
                try:
                    attrs = line.split(separator)[0:-1]
                    if row_attributes:
                        attrs = mcol(numpy.array(attrs,dtype=numpyDataType))
                    else:
                        attrs = numpy.array(attrs,dtype=numpyDataType)
                    label = line.split(separator)[-1].strip()
                    data.append(attrs)
                    labels.append(label)
                except:
                    pass
            if row_attributes:
                return numpy.hstack(data), numpy.array(labels)
            else:
                return numpy.array(data), numpy.array(labels)
        else:
            for line in f:
                try:
                    attrs = line.strip().split(separator)
                    if row_attributes:
                        attrs = mcol(numpy.array(attrs,dtype=numpyDataType))
                    else:
                        attrs = numpy.array(attrs,dtype=numpyDataType)
                    data.append(attrs)
                except:
                    pass
            return numpy.array(data)
        

def loadEncodedData(filename, row_attributes=True, numpyDataType=numpy.float32):
    D,L=loadData(filename, row_attributes=row_attributes, labels=True, numpyDataType=numpyDataType)
    L, L_names=oneHotEncoding(L, return_dictionary=True)
    return D,L,L_names

def read_txt(filename):
    with open(filename) as f:
        lines = f.readlines()
        return map(lambda x:x.strip(),lines)
    
def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris() ['target']
    return D, L

def db_train_test_split(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    x_train = D[:, idxTrain]
    x_test = D[:, idxTest]
    y_train = L[idxTrain]
    y_test = L[idxTest]
    return (x_train, y_train), (x_test, y_test)