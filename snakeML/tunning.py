from snakeML import SVM, logistic_regression, validation, gaussian
import matplotlib.pyplot as plt
import numpy as np


def testing_models(
    model, submodels, hyperparameter_values=None, hyperparameter_index=None
):
    """
    Generates a list of submodels to be tested, for a set of hyperparameters
    returns: list of submodels to be tested
        Example: [("linear", lambda), ("quadratic", lambda)]

    Parameters
    ----------
    model: model to be tested
    submodels: list of submodels to be tested without the hyperparameter
        Example: [("linear"), ("quadratic")]
    hyperparameter_values: list of hyperparameters to be tested
    hyperparameter_index: index of the hyperparameter in the submodel
        - Example: for logistic regression, hyperparameter_index=1 because the submodel is ("linear", lambda)
    """
    new_submodels = []
    if type(model) == SVM.SVM:
        if not hyperparameter_index:
            hyperparameter_index = 1
        if not hyperparameter_values:
            hyperparameter_values = [
                0.00001,
                0.0001,
                0.001,
                0.01,
                0.1,
                1,
                10,
                100,
                1000,
            ]
    elif type(model) == logistic_regression.logisticRegression:
        if not hyperparameter_values:
            hyperparameter_values = [
                10**-5,
                10**-4,
                10**-3,
                10**-2,
                10**-1,
                0,
                10**1,
                10**2,
            ]
    elif type(model) == gaussian.generativeClassifier:
        if not hyperparameter_values:
            tar_k = [2, 4, 8, 16]
            ntar_k = [2, 4, 8]
            hyperparameter_values = []
            for i in ntar_k:
                for j in tar_k:
                    hyperparameter_values.append([i, j])
        else:
            tar_k = []
            ntar_k = []
            for i in hyperparameter_values:
                if i[1] not in tar_k:
                    tar_k.append(i[1])
                if i[0] not in ntar_k:
                    ntar_k.append(i[0])
    else:
        raise Exception("Model not supported")
    for i in submodels:
        args = i[1]
        for j in hyperparameter_values:
            if hyperparameter_index:
                if type(args) is list:
                    new_args = (
                        args[0:hyperparameter_index] + [j] + args[hyperparameter_index:]
                    )
                else:
                    new_args = [args]
                    new_args.insert(hyperparameter_index, j)
                new_submodels.append((i[0], new_args))
            else:
                new_submodels.append((i, j))
    if type(model) == gaussian.generativeClassifier:
        return new_submodels, hyperparameter_values, tar_k, ntar_k
    else:
        return new_submodels, hyperparameter_values


def hyperparameter_search(
    D,
    L,
    model,
    submodels,
    metric,
    k=5,
    PCA=[None],
    Pc=[None],
    znorm=0,
    hyperparameter_values=None,
    title="Hyperparameter Search",
    parameter="Variable",
    plot=True,
    filename="results.xlsx",
):
    """
    Computes the avgMinDCF for a set of hyperparameters and plots the results
    returns: list of errors for each model
        - Shape: (len(PCAm)*len(Pc)*len(submodels), len(hyperparameter_values))

    Parameters
    ----------
    D: data matrix
    L: labels
    model: model to be trained
    submodels: list of submodels to be trained without the hyperparameter
        Example: [("linear"), ("quadratic")]
    k: number of folds
        K ---> k=1: LOO or k>1
    PCA: number of components for PCA
    metric: metric to be used
    title: title of the plot
    parameter: name of the hyperparameter
    """
    if type(model) == gaussian.generativeClassifier:
        if znorm == 1:
            znorm = 0
            print("znorm not supported for GMM, setting znorm=0")
        if len(PCA) > 1:
            PCA = PCA[0]
            print("Only one PCA supported for GMM, setting PCA=PCA[0]")
        new_submodels, hyperparameter_values, tar_k, ntar_k = testing_models(
            model, submodels, hyperparameter_values=hyperparameter_values
        )
    else:
        new_submodels, hyperparameter_values = testing_models(
            model, submodels, hyperparameter_values=hyperparameter_values
        )
    results = validation.kfold(
        D,
        L,
        model,
        new_submodels,
        k=k,
        PCAm=PCA,
        Pc=Pc,
        metric=metric,
        znorm=znorm,
        filename=filename + ".xlsx",
    )
    np.save("temporary.npy", results)
    if plot:
        if type(model) == gaussian.generativeClassifier:
            plot_kfold_hyper_gmm(
                results,
                hyperparameter_values,
                submodels=submodels,
                tar_k=tar_k,
                ntar_k=ntar_k,
                filename=filename,
            )
        else:
            plot_kfold_hyper(
                results,
                hyperparameter_values,
                submodels,
                filename=filename,
                PCAm=PCA,
                title=title,
                parameter=parameter,
                znorm=znorm,
            )


def plot_kfold_hyper(
    results,
    hyperparameter_values,
    submodels,
    PCAm,
    znorm,
    filename,
    save=True,
    title="Hyperparameter Search",
    parameter="Variable",
):
    """
    Plots the results of the hyperparameter search

    Parameters
    ----------
    results: list of errors for each model
        - Example: [[model1_error1, model1_error2, ...], ...]
    hyperparameter_values: list of hyperparameters to be tested
    submodels: list of submodels to be trained without the hyperparameter
        Example: [("linear"), ("quadratic")]
    save: boolean to save the plot
    title: title of the plot
    parameter: name of the hyperparameter
    """
    np.save(filename + ".npy", results)
    plt.figure()
    plt.grid()
    plt.xlabel(parameter)
    plt.ylabel("avg minDCF")
    plt.title(title)
    # ax.xaxis.set_ticks(range(len(lambdas))) #set the ticks to be a
    plt.xticks(
        range(len(hyperparameter_values)), [str(x) for x in hyperparameter_values]
    )
    iter = 0
    for j in range(len(PCAm)):
        for k in range(znorm + 1):
            for i in range(int(results.shape[1] / len(hyperparameter_values))):
                if type(submodels[i]) is tuple:
                    if type(submodels[i][1]) is list:
                        name = (
                            str(submodels[i][0])
                            + " "
                            + str(submodels[i][1])
                            + " | PCA:"
                            + str(PCAm[j])
                        )
                    else:
                        name = (
                            str(submodels[i][0])
                            + " ["
                            + str(submodels[i][1])
                            + "] | PCA:"
                            + str(PCAm[j])
                        )
                else:
                    name = submodels[i] + "| PCA:" + str(PCAm[j])
                if k:
                    name += "| Znorm"
                plt.plot(
                    range(len(hyperparameter_values)),
                    results[iter].flatten()[
                        i * len(hyperparameter_values) : i * len(hyperparameter_values)
                        + len(hyperparameter_values)
                    ],
                    label=name,
                )
            iter += 1
    plt.legend()
    if save:
        plt.savefig(filename + ".png")
    # plt.show()


def plot_kfold_hyper_gmm(
    results, hyperparameter_values, tar_k, ntar_k, filename, submodels, save=True
):
    np.save(filename + ".npy", results)
    results = results.round(3)
    results = results.reshape(len(submodels), len(hyperparameter_values))
    for m in range(len(submodels)):
        res = results[0][
            m * len(submodels) : m * len(submodels) + len(hyperparameter_values)
        ]
        values = {}
        for i in range(len(ntar_k)):
            values["Non-target K=" + str(ntar_k[i])] = ()
            for j in range(len(hyperparameter_values)):
                if hyperparameter_values[j][0] == ntar_k[i]:
                    val = results[m, j]
                    values["Non-target K=" + str(ntar_k[i])] = values[
                        "Non-target K=" + str(ntar_k[i])
                    ] + (val,)

        x = np.arange(len(tar_k))  # the label locations
        width = 0.25  # the width of the bars
        x = x * (len(ntar_k) + 1) * width
        multiplier = 1

        fig, ax = plt.subplots(layout="constrained")
        my_cmap = plt.get_cmap("gist_heat")

        for attribute, measurement in values.items():
            offset = width * multiplier
            rects = ax.bar(
                x + offset,
                measurement,
                width,
                label=attribute,
                color=my_cmap((multiplier - 1) / len(values)),
            )
            ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel("avg minDCF")
        ax.set_title(submodels[m])
        ax.set_xlabel("Target K")
        ticks = x + width / 2 + (width * len(ntar_k) / 2)
        ax.set_xticks(ticks, tar_k)
        ax.legend()
        ax.set_ylim(0, np.max(results[m]) + 0.05)
        if save:
            plt.savefig("results/GMM/" + submodels[m] + ".png")
        plt.show()
