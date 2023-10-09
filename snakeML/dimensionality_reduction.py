from matplotlib import pyplot as plt
from snakeML.numpy_transformations import mcol
import numpy
import scipy
from snakeML.visualization import (
    scatter_attributeVSattribute,
    histogram_attributeVSfrequency,
)


def PCA(x_train, m, x_test=[], return_transformation=False, flipped=False):
    """
    Principal Component Analysis
    returns: transformed x_train matrix, transformed x_test matrix (optional), transformation matrix (optional)

    Parameters
    ----------
    x_train: data matrix
    m: number of dimensions after transformation
    x_test: test data matrix (optional)
    return_transformation: if True, the transformation matrix is returned
    flipped: if True, the sign of the eigenvectors is flipped
    """
    mu = x_train.mean(1)
    DC = x_train - mcol(mu)
    C = numpy.dot(DC, DC.T) / x_train.shape[1]
    s, U = numpy.linalg.eigh(C)
    if flipped:
        P = -U[:, ::-1][:, 0:m]
    else:
        P = U[:, ::-1][:, 0:m]
    DPtrain = numpy.dot(P.T, x_train)
    if len(x_test) > 0:
        DPtest = numpy.dot(P.T, x_test)
        if return_transformation:
            return DPtrain, DPtest, P
        else:
            return DPtrain, DPtest
    else:
        if return_transformation:
            return DPtrain, P
        else:
            return DPtrain


def scatter_PCA(
    D,
    L,
    L_names,
    m,
    x_test=[],
    flipped=False,
    folder="",
    save_image=False,
    img_name="PCA",
):
    """
    Generates class distributions for each combination of two attributes with PCA.
    returns: transformed x_train matrix, transformed x_test matrix (optional) and saves png image in "path"

    Parameters
    ----------
    D: data matrix
    L: labels
    L_names: labels names dictionary
        Example: {0: "class 0", 1: "class 1"}
    m: number of dimensions after transformation
    x_test: test data matrix (optional)
    flipped: if True, the sign of the eigenvectors is flipped
    folder: folder to save the image
    save_image: if True, the image is saved
    img_name: name of the image
    """
    if len(x_test) > 0:
        DPtrain, DPtest = PCA(D, m, flipped=flipped, x_test=x_test)
    else:
        DPtrain = PCA(D, m, flipped=flipped)
    features = []
    for i in range(m):
        features.append("f" + str(i))
    scatter_attributeVSattribute(
        DPtrain,
        L,
        features,
        L_names,
        row_attributes=True,
        is_label_dict=True,
        name=img_name,
        folder=folder,
        save=save_image,
    )
    if len(x_test) > 0:
        return DPtrain, DPtest
    else:
        return DPtrain


def scatter_histogram_PCA(
    D,
    L,
    L_names,
    m,
    x_test=[],
    flipped=False,
    folder="",
    save_image=False,
    img_name="PCA",
):
    """
    Generates scatter plot of class distributions for each combination of two attributes with PCA.
    Generates histogram of class distributions for each attribute.
    returns: transformed x_train matrix, transformed x_test matrix (optional) and saves png image in "path"

    Parameters
    ----------
    PCA: PCA dimensions
    """
    if len(x_test) > 0:
        DPtrain, DPtest = PCA(D, m, flipped=flipped, x_test=x_test)
    else:
        DPtrain = PCA(D, m, flipped=flipped)
    features = []
    for i in range(m):
        features.append("f" + str(i))
    histogram_attributeVSfrequency(
        DPtrain,
        L,
        features,
        L_names,
        is_label_dict=True,
        row_attributes=True,
        dense=True,
        center_data=True,
        save=save_image,
        colors=["royalblue", "limegreen"],
        folder=folder,
        name=img_name,
    )
    scatter_attributeVSattribute(
        DPtrain,
        L,
        features,
        L_names,
        row_attributes=True,
        is_label_dict=True,
        name=img_name,
        folder=folder,
        save=save_image,
    )
    if len(x_test) > 0:
        return DPtrain, DPtest
    else:
        return DPtrain


def LDA(x_train, y_train, m, x_test=None, return_transformation=False):
    """
    Linear Discriminant Analysis
    returns: transformed x_train matrix, transformed x_test matrix (optional), transformation matrix (optional)

    Parameters
    ----------
    x_train: data matrix
    y_train: labels
    m: number of dimensions after transformation
    x_test: test data matrix (optional)
    return_transformation: if True, the transformation matrix is returned
    """
    mu = x_train.mean(1)
    SW = numpy.zeros(x_train.shape[0])
    SB = numpy.zeros(x_train.shape[0])
    labels = numpy.unique(y_train)
    for i in labels:
        # SW
        data = x_train[:, y_train == labels[i]]
        muC = data.mean(1)
        DC = data - mcol(muC)
        CW = numpy.dot(DC, DC.T)
        SW = SW + CW
        # SB
        DM = mcol(muC) - mcol(mu)
        CB = numpy.dot(DM, DM.T) * data.shape[1]
        SB = SB + CB
    SW = SW / x_train.shape[1]
    SB = SB / x_train.shape[1]
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]
    UW, _, _ = numpy.linalg.svd(W)
    U = UW[:, 0:m]
    DPtrain = numpy.dot(W.T, x_train)
    if x_test:
        DPtest = numpy.dot(W.T, x_test)
        if return_transformation:
            return DPtrain, DPtest, W
        else:
            return DPtrain, DPtest
    else:
        if return_transformation:
            return DPtrain, W
        else:
            return DPtrain


def scatter_LDA(
    D,
    L,
    L_names,
    m,
    x_test=[],
    flipped=False,
    folder="",
    save_image=False,
    img_name="LDA",
):
    """
    Generates class distributions for each combination of two attributes with LDA.
    returns: transformed x_train matrix, transformed x_test matrix (optional) and saves png image in "path"

    Parameters
    ----------
    D: data matrix
    L: labels
    L_names: labels names dictionary
        Example: {0: "class 0", 1: "class 1"}
    m: number of dimensions after transformation
    x_test: test data matrix (optional)
    flipped: if True, the sign of the eigenvectors is flipped
    folder: folder to save the image
    save: if True, the image is saved
    img_name: name of the image
    """
    if len(x_test) > 0:
        DPtrain, DPtest = LDA(D, L, L_names, m, flipped)
    else:
        DPtrain = LDA(D, L, L_names, m, flipped)
    features = []
    for i in range(m):
        features.append("f" + str(i))
    scatter_attributeVSattribute(
        DPtrain,
        L,
        features,
        L_names,
        row_attributes=True,
        is_label_dict=True,
        name=img_name,
        folder=folder,
        save=save_image,
    )
    if len(x_test) > 0:
        return DPtrain, DPtest
    else:
        return DPtrain


def scatter_histogram_LDA(
    D,
    L,
    L_names,
    m,
    x_test=[],
    flipped=False,
    folder="",
    save_image=False,
    img_name="LDA",
):
    """
    Generates scatter plot of class distributions for each combination of two attributes with LDA.
    Generates histogram of class distributions for each attribute.
    returns: transformed x_train matrix, transformed x_test matrix (optional) and saves png image in "path"

    Parameters
    ----------
    D: data matrix
    L: labels
    L_names: labels names
    m: number of dimensions after transformation
    x_test: test data matrix (optional)
    flipped: if True, the sign of the eigenvectors is flipped
    folder: folder to save the image
    save_image: if True, the image is saved
    img_name: name of the image
    """
    if len(x_test) > 0:
        DPtrain, DPtest = LDA(D, L, m, x_test=x_test)
    else:
        DPtrain = LDA(D, L, m)
    features = []
    for i in range(m):
        features.append("f" + str(i))
    histogram_attributeVSfrequency(
        DPtrain,
        L,
        features,
        L_names,
        is_label_dict=True,
        row_attributes=True,
        dense=True,
        center_data=True,
        save=save_image,
        colors=["royalblue", "limegreen"],
        folder=folder,
        name=img_name,
    )
    scatter_attributeVSattribute(
        DPtrain,
        L,
        features,
        L_names,
        row_attributes=True,
        is_label_dict=True,
        name=img_name,
        folder=folder,
        save=save_image,
    )
    if len(x_test) > 0:
        return DPtrain, DPtest
    else:
        return DPtrain


def PCA_variance_plot(x_train, feature_num, save=False, folder="", name=""):
    """
    Plots PCA (# of features) vs. Variance (%)

    Parameters
    ----------
    x_train: data matrix
    feature_num: number of features
    """
    mu = x_train.mean(1)
    DC = x_train - mcol(mu)
    C = numpy.dot(DC, DC.T) / x_train.shape[1]
    s, U = numpy.linalg.eigh(C)
    vars = [0]
    sum = 0
    features = numpy.arange(feature_num + 1)
    for i in range(feature_num):
        sum += s[::-1][i]
        vars.append(sum)
    vars = numpy.array(vars)
    vars = vars / s.sum()
    plt.plot(features, vars)
    plt.grid()
    plt.xticks(features, features)
    plt.yticks(numpy.arange(0, 1.1, 0.05))
    plt.xlabel("Number of features")
    plt.ylabel("Variance")
    plt.show()
    if save:
        path = folder + "PCA_variance" + name + ".png"
        plt.savefig(path)
        print(path, " saved")
