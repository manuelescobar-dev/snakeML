from snakeML.loads import loadData
from snakeML.numpy_transformations import mcol
from snakeML.visualization import scatter_attributeVSattribute
import numpy
import scipy

def PCA(D, m, flipped=False):
    mu=D.mean(1)
    DC=D-mcol(mu)
    C=numpy.dot(DC,DC.T)/D.shape[1]
    s, U = numpy.linalg.eigh(C)
    if flipped:
        P = -U[:, ::-1][:, 0:m]
    else:
        P = U[:, ::-1][:, 0:m]
    DP=numpy.dot(P.T,D)
    return DP

def scatter_PCA(D,L,L_names,m,flipped=False):
    #D,L=loadData(filename, row_attributes=True, labels=True, numpyDataType=numpy.float32)
    #L, L_names=oneHotEncoding(L, return_dictionary=True)
    DP=PCA(D,m, flipped)
    features=[]
    for i in range(m):
        features.append("Axis "+str(i))
    scatter_attributeVSattribute(DP,L,features,L_names,row_attributes=True,is_label_dict=True)

def LDA(D, L,L_names, m,flipped=False):
    mu=D.mean(1)
    SW=numpy.zeros(D.shape[0])
    SB=numpy.zeros(D.shape[0])
    for i in L_names:
        #SW
        data=D[:, L==L_names[i]]
        muC=data.mean(1)
        DC=data-mcol(muC)
        C=numpy.dot(DC,DC.T)
        SW=SW+C
        #SB
        DC=mcol(muC)-mcol(mu)
        C=numpy.dot(DC,DC.T)*data.shape[1]
        SB=SB+C
    SW=SW/D.shape[1]
    SB=SB/D.shape[1]
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]
    UW, _, _ = numpy.linalg.svd(W)
    U = UW[:, 0:m]
    DP=numpy.dot(W.T,D)
    return DP

def scatter_LDA(D, L, L_names, m,flipped=False):
    DP=LDA(m,D,L, L_names, flipped)
    features=[]
    for i in range(m):
        features.append("Axis "+str(i))
    scatter_attributeVSattribute(DP,L,features,L_names,row_attributes=True,is_label_dict=True)

