import math
import matplotlib.pyplot as plt
from snakeML.numpy_transformations import mcol, mrow
import numpy

# Graph proportionality (default for documents)
graph_prop = [6, 4]
# Maximum figures per row
fig_max_width = 3


def find_dimensions(total, max_width=fig_max_width):
    """
    Finds the dimensions of a grid of subplots.
    returns: rows, columns

    Parameters
    ----------
    total: number of subplots
    max_width: maximum number of subplots per row
    """
    cols = max_width
    rows = math.ceil(total / cols)
    return rows, cols


def histogram_attributeVSfrequency(
    data,
    labels,
    features,
    label_names,
    is_label_dict=False,
    row_attributes=False,
    dense=False,
    save=False,
    center_data=True,
    colors=[None, None],
    folder="",
    show=False,
    name="",
):
    """
    Plots histograms of the frequency of each attribute for each class.

    Parameters
    ----------
    data: data matrix
    labels: labels
    features: list of attribute names
    label_names: dictionary of label names
    is_label_dict: if True, label_names is a dictionary
    row_attributes: if True, each column represents a sample and each row an attribute
    dense: if True, the histogram is normalized
    save: if True, the image is saved
    center_data: if True, the data is centered
    colors: list of colors for each class
    folder: folder to save the image
    show: if True, the image is shown
    name: name of the image
    """
    rows, cols = find_dimensions(len(features))
    plt.figure(figsize=(cols * graph_prop[0], rows * graph_prop[1]), dpi=200)
    if center_data:
        if row_attributes:
            data = data - mcol(data.mean(axis=1))
        else:
            data = data - mrow(data.mean(axis=0))
    if is_label_dict:
        lab = list(label_names.keys())
    else:
        lab = label_names
    for i in range(len(features)):
        plt.subplot(rows, cols, i + 1)
        plt.xlabel(features[i])
        for j in range(len(lab)):
            if row_attributes:
                plt.hist(
                    data[:, labels == j][i, :],
                    density=dense,
                    label=lab[j],
                    color=colors[0] if j == 0 else colors[1],
                    alpha=0.65,
                    bins=50,
                    edgecolor="white",
                    linewidth=0.5,
                )
            else:
                plt.hist(
                    data[labels == j, :][:, i],
                    density=dense,
                    label=lab[j],
                    color=colors[0] if j == 0 else colors[1],
                    alpha=0.65,
                    bins=50,
                    edgecolor="white",
                    linewidth=0.5,
                )
        plt.legend()
        plt.tight_layout()
    if save:
        path = (
            folder
            + "hist"
            + ("_dense" if dense else "_notdense")
            + ("_centered" if center_data else "_notcentered")
            + name
            + ".png"
        )
        plt.savefig(path)
        print(path, " saved")
    if show:
        plt.show()


def scatter_attributeVSattribute(
    data,
    labels,
    features,
    label_names,
    is_label_dict=False,
    row_attributes=False,
    save=False,
    center_data=False,
    colors=[None, None],
    folder="",
    show=False,
    columns=fig_max_width,
    name="",
):
    """
    Plots scatter plots of each combination of two attributes for each class.

    Parameters
    ----------
    data: data matrix
    labels: labels
    features: list of attribute names
    label_names: dictionary of label names
    is_label_dict: if True, label_names is a dictionary
    row_attributes: if True, each column represents a sample and each row an attribute
    save: if True, the image is saved
    center_data: if True, the data is centered
    colors: list of colors for each class
    folder: folder to save the image
    show: if True, the image is shown
    columns: maximum number of subplots per row
    name: name of the image
    """
    rows, cols = find_dimensions(
        (((len(features) - 1) * len(features)) / 2) if len(features) > 1 else 1,
        max_width=columns,
    )
    plt.figure(figsize=(cols * graph_prop[0], rows * graph_prop[1]), dpi=200)
    if center_data:
        if row_attributes:
            data = data - mcol(data.mean(axis=1))
        else:
            data = data - mrow(data.mean(axis=0))
    if is_label_dict:
        lab = list(label_names.keys())
    else:
        lab = label_names
    counter = 1
    for i in range(len(features)):
        for k in range(len(features)):
            if i >= k:
                continue
            plt.subplot(rows, cols, counter)
            counter += 1
            plt.xlabel(features[i])
            plt.ylabel(features[k])
            for j in range(len(lab)):
                if row_attributes:
                    plt.scatter(
                        data[:, labels == j][i, :],
                        data[:, labels == j][k, :],
                        label=lab[j],
                        color=colors[0] if j == 0 else colors[1],
                    )
                else:
                    plt.scatter(
                        data[labels == j, :][:, i],
                        data[labels == j, :][:, k],
                        label=lab[j],
                        color=colors[0] if j == 0 else colors[1],
                    )
            plt.legend()
            plt.tight_layout()
    if show:
        plt.show()
    if save:
        path = folder + "scatter" + name + ".png"
        plt.savefig(folder + "scatter" + name + ".png")
        print(path, "saved")


def scatter_categories(
    data, labels, label_names, is_label_dict=False, row_attributes=False, save=False
):
    """
    Plots scatter plots of each combination of two attributes for each class.
    """
    if is_label_dict:
        lab = list(label_names.keys())
    else:
        lab = label_names
    plt.figure()
    for j in range(len(lab)):
        if row_attributes:
            plt.scatter(data[:, labels == j], data[:, labels == j], label=lab[j])
        else:
            plt.scatter(data[labels == j, :], data[labels == j, :], label=lab[j])
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig("scatter_%d_%d.png")
    plt.show()


def pearson_correlation_binary(data, labels):
    """
    Pearson correlation for each class.

    Parameters
    ----------
    data: data matrix
    labels: labels
    """
    plt.subplot(1, 3, 1)
    cor = numpy.abs(numpy.corrcoef(data[:, labels == 0], y=None, rowvar=True))
    im = plt.imshow(cor, cmap="gray_r", interpolation="nearest")
    plt.colorbar(im)
    plt.title("Male")

    plt.subplot(1, 3, 2)
    cor = numpy.abs(numpy.corrcoef(data[:, labels == 1], y=None, rowvar=True))
    im = plt.imshow(cor, cmap="hot_r", interpolation="nearest")
    plt.colorbar(im)
    plt.title("Female")

    plt.subplot(1, 3, 3)
    cor = numpy.abs(numpy.corrcoef(data, y=None, rowvar=True))
    im = plt.imshow(cor, cmap="winter_r", interpolation="nearest")
    plt.colorbar(im)
    plt.title("Dataset")
    plt.show()


def logpdf_GAU_ND_visualization(Data, result):
    """
    Plots the logpdf of a Gaussian distribution.

    Parameters
    ----------
    Data: data matrix
    result: result of the logpdf
    """
    plt.figure()
    plt.plot(Data.ravel(), numpy.exp(result))
    plt.show()


def loglikelihood_visualization(Data, XPlot, Result):
    """
    Plots the loglikelihood of a Gaussian distribution.

    Parameters
    ----------
    Data: data matrix
    XPlot: data matrix to plot
    Result: result of the logpdf
    """
    plt.figure()
    plt.hist(Data.ravel(), bins=50, density=True)
    plt.plot(XPlot.ravel(), numpy.exp(Result))
    plt.show()
