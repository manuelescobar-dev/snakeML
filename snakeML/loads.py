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
        

def loadEncodedData(filename):
    D,L=loadData(filename, row_attributes=True, labels=True, numpyDataType=numpy.float32)
    L, L_names=oneHotEncoding(L, return_dictionary=True)
    return D,L,L_names