from snakeML import validation, preprocessing, metrics, logistic_regression
from snakeML.dimensionality_reduction import PCA
import numpy

# Currently under development


class Fusion:
    """
    Fusion model class used for training and evaluating a fused model.

    Attributes
    ----------
    """

    def __init__(self, models, submodels, model_names):
        """
        Parameters
        ----------
        models : list
            List of models to be fused.
        submodels : list
            List of submodels to be fused.
        model_names : list
            List of names for each model.
        """
        self.models = models
        self.submodels = submodels
        self.model_names = model_names

    def train(self, xtrain, ytrain):
        """
        Trains the fused model.

        Parameters
        ----------
        xtrain : numpy.ndarray
            Training set.
        ytrain : numpy.ndarray
            Training labels.
        """
        if len(self.models) > 2:
            raise Exception("Fusion only works with two models")
        else:
            # Model 1
            score = self.models[0].train(
                xtrain, ytrain, submodels=self.submodels[0], return_score=True
            )
            score = score.reshape((1, -1))
            # Model 2
            self.models[1].train(score, ytrain, submodels=self.submodels[1])

    def evaluate(self, xtest, ytest, metric=["Accuracy"], return_score=False):
        """
        Evaluates the fused model.

        Parameters
        ----------
        xtest : numpy.ndarray
            Test set.
        ytest : numpy.ndarray
            Test labels.
        metric : list, optional
            List of metrics to be evaluated. The default is ["Accuracy"].
        return_score : bool, optional
        """
        # Model 1
        score = self.models[0].evaluate(
            xtest, ytest, submodels=self.submodels[0], return_score=True
        )
        score = score.reshape((1, -1))
        # Model 2
        pred, m = self.models[1].evaluate(score, ytest, metric=metric)
        if return_score:
            return m, score
        else:
            return m


def score_calibration(
    D,
    L,
    Rk_models,
    Ck_models,
    Rk_submodels,
    Ck_submodels,
    model_names,
    Pt=None,
    filename="calibration",
    plot_bayes=True,
):
    results = []
    for i in range(len(Rk_models)):
        for j in range(len(Ck_models)):
            # Trains over each fold of the dataset
            scores = kfold_calibration(D, L, Rk_models[i], Rk_submodels[i])
            scores = scores.reshape((1, -1))
            # Trains over the whole dataset
            # Rf.train(D, L, submodels)

            # Trains over each fold of the score set
            calibrated_score = kfold_calibration(
                scores, L, Ck_models[j], Ck_submodels[j], seed=1
            )
            if Pt is not None:
                calibrated_score = calibrated_score - numpy.log(Pt / (1 - Pt))
            results.append(calibrated_score)

            numpy.save(filename + ".npy", results)
            # Trains over the whole score set

    if plot_bayes:
        metrics.bayes_error_plot(
            results,
            L,
            submodels=model_names,
            effPriorLB=-4,
            effPriorUB=4,
            precision=0.1,
            filename=filename,
        )

    # Cf.train(scores, L, submodels)
    return results


def model_fusion(
    D,
    L,
    model_names,
    Ck=logistic_regression.logisticRegression(),
    Ck_submodels=[("linear", 0.001)],
    return_scores=True,
    metric=["Accuracy"],
    models=None,
    submodels=None,
    models_scores=None,
    Pt=None,
    filename="fusion",
    plot_bayes=True,
):
    results = []
    if models_scores is None:
        models_scores = []
        if models is None or submodels is None:
            raise Exception("No models or scores were given")
        else:
            for i in range(len(models)):
                s = kfold_calibration(D, L, models[i], submodels[i])
                s = s.reshape((1, -1))
                models_scores.append(kfold_calibration(D, L, models[i], submodels[i]))

    models_scores = numpy.array(models_scores)

    if return_scores:
        # Three models
        training_scores = models_scores
        calibrated_score = kfold_calibration(
            training_scores, L, Ck, Ck_submodels, seed=1
        )
        if Pt is not None:
            calibrated_score = calibrated_score - numpy.log(Pt / (1 - Pt))
        results.append(calibrated_score)

        # Combinations
        for i in range(len(models_scores)):
            for j in range(len(models_scores)):
                if i < j:
                    training_scores = models_scores[[i, j], :]
                    # Trains over each fold of the score set
                    calibrated_score = kfold_calibration(
                        training_scores, L, Ck, Ck_submodels, seed=1
                    )
                    if Pt is not None:
                        calibrated_score = calibrated_score - numpy.log(Pt / (1 - Pt))
                    results.append(calibrated_score)
    else:
        # Three models
        training_scores = models_scores
        m = validation.kfold(
            training_scores, L, Ck, Ck_submodels, seed=1, save=False, metric=metric
        )
        m = m.round(3)
        results.append(m)

        if plot_bayes:
            metrics.bayes_error_plot(
                results,
                L,
                submodels=model_names,
                effPriorLB=-4,
                effPriorUB=4,
                precision=0.25,
                filename=filename,
            )

        # Combinations
        for i in range(len(models_scores)):
            for j in range(len(models_scores)):
                if i < j:
                    training_scores = models_scores[[i, j], :]
                    # Trains over each fold of the score set
                    m = validation.kfold(
                        training_scores,
                        L,
                        Ck,
                        Ck_submodels,
                        save=False,
                        seed=1,
                        metric=metric,
                    )
                    m = m.round(3)
                    results.append(m)

    return results


def kfold_calibration(D, L, model, submodels, k=5, PCAm=None, znorm=0, seed=0):
    indexes = numpy.arange(D.shape[1])
    scores = []

    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    folds = []
    scores = numpy.zeros(len(L))
    for i in range(k):
        folds.append(idx[int(i * (D.shape[1] / k)) : int((i + 1) * (D.shape[1] / k))])
    for i in range(len(folds)):
        c = numpy.in1d(indexes, folds[i])
        cinv = numpy.invert(c)
        x_train = D[:, cinv]
        x_test = D[:, c]
        y_train = L[cinv]
        y_test = L[c]
        if PCAm is not None:
            xtraining, xtesting = PCA(x_train, PCAm, x_test=x_test)
        else:
            xtraining, xtesting = x_train, x_test
        if znorm == 1:
            xtraining, xtesting = preprocessing.znorm(xtraining, xtesting)
        """ if Pc=="Fold":
            Pc=[y_train.sum()/len(y_train)]
        elif Pc=="Dataset":
            Pc=[L.sum()/len(L)] """
        model.train(xtraining, y_train, submodels=submodels)
        score = model.evaluate(xtesting, y_test, return_score=True)
        score = numpy.array(score[0])
        if score.ndim > 1:
            if score.shape[1] > 1 and score.shape[0] > 1:
                score = score[1, :] - score[0, :]
            else:
                score = score.flatten()
        scores[c] = score
        print("Fold: ", i + 1, "/", k, end="\r")
    return scores


def calibrated(
    xtrain,
    ytrain,
    xtest,
    ytest,
    model,
    submodel,
    Ck=logistic_regression.logisticRegression(),
    Ck_submodels=[("linear", 0.001)],
    return_scores=True,
    metric=["Accuracy"],
    models=None,
    submodels=None,
    models_scores=None,
    Pt=None,
    filename="fusion",
    plot_bayes=True,
):
    # ----------------------------------Before fusion----------------------------------
    model.train(xtrain, ytrain, submodels=[submodel])
    s = model.evaluate(xtest, ytest, metric=metric, return_scores=True)
    s = s.reshape((1, -1))

    # ----------------------------------Fusion----------------------------------
    # Three models
    Ck.train(s, ytest, submodels=Ck_submodels)
    pred, m = Ck.evaluate(s, ytest, metric=metric)

    if plot_bayes:
        scores = Ck.evaluate(s, ytest, metric=metric, return_scores=True)
        numpy.save("scores" + ".npy", scores)
        metrics.bayes_error_plot(
            scores,
            ytest,
            submodels=[submodel],
            effPriorLB=-4,
            effPriorUB=4,
            precision=0.25,
            filename=filename,
        )
        return scores, m

    return m


def calibration_split(xtrain, ytrain, calibration_split=0.2, seed=0):
    nTrain = int(xtrain.shape[1] * calibration_split)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(xtrain.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    x_train = xtrain[:, idxTrain]
    x_cal = xtrain[:, idxTest]
    y_train = ytrain[idxTrain]
    y_cal = ytrain[idxTest]
    return (x_train, y_train), (x_cal, y_cal)
