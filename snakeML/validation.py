import numpy
from snakeML.dimensionality_reduction import PCA
from snakeML import validation, preprocessing
import pandas as pd


def kfold(
    D,
    L,
    model,
    submodels,
    metric,
    k=5,
    PCAm=[None],
    Pc=[None],
    seed=0,
    save=True,
    znorm=0,
    filename="results.xlsx",
):
    """
    Kfold validation
    returns: list of errors for each model
        - Example: [[model1_error1, model1_error2, ...], ...]

    Parameters
    ----------
    D: data matrix
    L: labels
    model: model to be trained
    submodels: list of submodels to be trained
    k: number of folds
        K ---> k=1: LOO or k>1
    PCAm: number of components for PCA
    """
    if not k > 0:
        print("Invalid K ---> k=1: LOO or k>1")
        return 0
    indexes = numpy.arange(D.shape[1])
    errors = []
    # Leave one out approach
    if k == 1:
        for i in range(D.shape[1]):
            y_train = L[indexes != i]
            y_test = L[indexes == i]
            x_train = D[:, indexes != i]
            x_test = D[:, indexes == i]
            for w in PCAm:
                if w:
                    xtraining, xtesting = PCA(x_train, w, x_test=x_test)
                    if znorm:
                        xtraining, xtesting = validation.znorm(xtraining, xtesting)
                else:
                    xtraining, xtesting = x_train, x_test
                    if znorm:
                        xtraining, xtesting = validation.znorm(xtraining, xtesting)
                model.train(xtraining, xtesting, models=submodels)
                pred, m = model.evaluate(x_test, y_test, metric=metric)
                errors.append(m)
    # Kfold approach
    else:
        numpy.random.seed(seed)
        idx = numpy.random.permutation(D.shape[1])
        folds = []
        for i in range(k):
            folds.append(
                idx[int(i * (D.shape[1] / k)) : int((i + 1) * (D.shape[1] / k))]
            )
        for i in range(len(folds)):
            c = numpy.in1d(indexes, folds[i])
            cinv = numpy.invert(c)
            x_train = D[:, cinv]
            x_test = D[:, c]
            y_train = L[cinv]
            y_test = L[c]
            row = 0
            for w in range(len(PCAm)):
                for j in range(znorm + 1):
                    for prior in Pc:
                        if PCAm[w]:
                            xtraining, xtesting = PCA(x_train, PCAm[w], x_test=x_test)
                        else:
                            xtraining, xtesting = x_train, x_test
                        if j:
                            xtraining, xtesting = preprocessing.znorm(
                                xtraining, xtesting
                            )
                        if prior == "Fold":
                            prior = [y_train.sum() / len(y_train)]
                        elif prior == "Dataset":
                            prior = [L.sum() / len(L)]
                        model.train(xtraining, y_train, submodels=submodels, Pc=prior)
                        pred, m = model.evaluate(xtesting, y_test, metrics_list=metric)
                        if i == 0:
                            errors.append(m)
                        else:
                            errors = numpy.array(errors)
                            errors[row] += m
                            row += 1
                print(
                    "Preprocessing: ",
                    w + 1,
                    "/",
                    len(PCAm),
                    " | Fold: ",
                    i + 1,
                    "/",
                    k,
                    end="\r",
                )
    errors = errors / k
    if save:
        save_results(
            submodels,
            metrics=metric,
            filename=filename,
            results=errors,
            PCA=PCAm,
            Pc=Pc,
            znorm=znorm,
        )
    return errors


def save_results(submodels, metrics, filename, results, PCA, Pc, znorm):
    """
    Saves results in excel file.
    """
    headers = ["Preprocessing"]
    params = [""]
    for i in submodels:
        if type(i) is tuple:
            model = i[0]
            par = i[1]
        else:
            model = i
            par = ""
        for h in metrics:
            if type(h) is tuple:
                metric = h[0]
                if metric == "minAndAvgDCF":
                    for n in range(len(h[1])):
                        headers.append(
                            model + " | " + metric + " (" + str(h[1][n][0]) + ")"
                        )
                        params.append(par)
                    headers.append(model + " | avg minDCF")
                    params.append(par)
                elif metric == "avgMinDCF":
                    headers.append(model + " | avg minDCF")
                    params.append(par)
            else:
                metric = h
                headers.append(model + " | " + h)
                params.append(par)
    rows = [params]
    rownum = 0
    for i in range(len(PCA)):
        for k in range(znorm + 1):
            for j in range(len(Pc)):
                if PCA[i]:
                    row = ["PCA (" + str(PCA[i]) + ")"]
                else:
                    row = ["- "]
                if Pc[j]:
                    row[0] += " | Prior: " + str(Pc[j])
                if k:
                    row[0] += " | Z-norm"
                row.extend(results[rownum].flatten())
                rownum += 1
                rows.append(row)
    df = pd.DataFrame(rows, columns=headers)
    df.to_excel(filename, index=False)


# Depreciated
def kfold_minDCF(
    D,
    L,
    models_submodels,
    metric,
    k=5,
    PCAm=None,
    seed=0,
    tune_hyperparameters=False,
    k_args=None,
):
    if not k > 0:
        print("Invalid K ---> k=1: LOO or k>1")
        return 0
    indexes = numpy.arange(D.shape[1])
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    err = numpy.zeros(len(metric[0][1]))
    errors = numpy.zeros((len(models_submodels), len(metric[0][1])))
    folds = []
    errors = []
    for i in range(k):
        folds.append(idx[int(i * (D.shape[1] / k)) : int((i + 1) * (D.shape[1] / k))])
    for i in range(len(folds)):
        c = numpy.in1d(indexes, folds[i])
        cinv = numpy.invert(c)
        x_train = D[:, cinv]
        x_test = D[:, c]
        y_train = L[cinv]
        y_test = L[c]
        if PCAm:
            x_train, x_test = PCA(x_train, PCAm, x_test=x_test)
        for j in range(len(models_submodels)):
            model = models_submodels[j][0]
            if tune_hyperparameters:
                if k_args:
                    lambdas, m = model.hyperparameter_tunning(
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        metric_list=metric,
                        binary=True,
                        models=models_submodels[j][1],
                        k_args=k_args,
                    )
                else:
                    lambdas, m = model.hyperparameter_tunning(
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        metric_list=metric,
                        binary=True,
                        models=models_submodels[j][1],
                    )
            else:
                model.train(x_train, y_train, models=models_submodels[j][1])
                pred, m = model.evaluate(x_test, y_test, metric_list=metric)
                m = numpy.array(m).reshape(
                    (len(models_submodels[j][1]), len(metric[0][1]))
                )
            if i == 0:
                errors.append(m)
            else:
                errors[j] += m
        print("Fold: ", i + 1, "/", k)
    errors = numpy.array(errors)
    errors = errors / k
    if tune_hyperparameters:
        avg = errors.mean(axis=3)
        return lambdas, avg
    else:
        avg = errors.mean(axis=2)
        avg = numpy.expand_dims(avg, axis=2)
        result = numpy.concatenate((errors, avg), axis=2)
        return result
