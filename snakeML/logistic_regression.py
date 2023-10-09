import numpy
from scipy.optimize import fmin_l_bfgs_b
from snakeML import metrics
import matplotlib.pyplot as plt


class logisticRegression:
    def __init__(self):
        self.params = []
        self.predictions = []
        self.metrics = []

    def train(
        self,
        xtrain,
        ytrain,
        submodels=[("linear", 0.001), ("quadratic", 0.001)],
        Pc=None,
    ):
        """
        Trains logistic regression models

        Parameters
        ----------
        xtrain: training data matrix
        ytrain: training labels
        submodels: list of submodels to be trained
            - ("linear", lambda)
            - ("quadratic", lambda)
        Pc: list of priors for each submodel
        """
        self.params = []
        self.predictions = []
        self.metrics = []
        self.models = submodels
        classes = numpy.unique(ytrain)
        if len(classes) == 2:
            binary = True
        else:
            binary = False
        for i in range(len(submodels)):
            if type(submodels[i]) is tuple:
                model = submodels[i][0]
                l = submodels[i][1]
            else:
                model = submodels[i]
            if model == "quadratic":
                quadratic = True
            else:
                quadratic = False
            if binary:
                self.isBinary = True
                if Pc:
                    w, b = self.binary_logRegression(
                        xtrain, ytrain, l, quadratic=quadratic, Pc=Pc
                    )
                else:
                    w, b = self.binary_logRegression(
                        xtrain, ytrain, l, quadratic=quadratic
                    )
                self.params.append((w, b))
            else:
                self.isBinary = False
                w, b = self.logRegression(xtrain, ytrain, l)
                self.params.append((w, b))

    def evaluate(self, xtest, ytest, metrics_list=["Accuracy"], return_score=False):
        """
        Evaluates the trained models on the test data
        returns: predictions, metrics
            - predictions: list of predictions for each model
            - metrics: list of metrics for each model
                Example: [[model1_metric1, model1_metric2, ...], [model2_metric1, model2_metric2, ...]...]

        Parameters
        ----------
        xtest: test data matrix
        ytest: test labels
        metrics_list: list of metrics to be computed
            - Accuracy
            - Error
            - DCF
            - minDCF
            - minAndAvgDCF
            - avgMinDCF
        """
        self.predictions = []
        self.metrics = []
        scores = []
        for i in range(len(self.params)):
            if self.models[i][0] == "quadratic":
                up = numpy.dot(xtest.T, xtest).diagonal().reshape((1, -1))
                x = numpy.vstack((up, xtest))
            else:
                x = xtest
            score = numpy.dot(self.params[i][0].T, x) + self.params[i][1]
            if self.isBinary:
                LP = []
                for i in score:
                    if i > 0:
                        LP.append(1)
                    else:
                        LP.append(0)
            else:
                LP = numpy.argmax(score, axis=0)
            if return_score:
                scores.append(score)
            else:
                self.predictions.append(LP)
                err = metrics.test_metrics(LP, ytest, metrics_list, args=score)
                # self.metrics.append(numpy.array(err))}
                self.metrics.append(err)
        if return_score:
            return scores
        else:
            return self.predictions, self.metrics

    # Binary logistic regression
    def binary_logreg_obj(self, v, xtrain, ytrain, l):
        w, b = v[0:-1], v[-1]
        zi = 2 * ytrain - 1
        p0 = numpy.linalg.norm(w)
        r0 = (l / 2) * numpy.power((p0), 2)
        p1 = numpy.dot(w.T, xtrain) + b
        p2 = numpy.multiply(-zi, p1)
        p3 = numpy.logaddexp(0, p2)
        r1 = p3.sum(axis=0) / xtrain.shape[1]
        return r0 + r1

    def prior_binary_logreg_obj(self, v, xtrain, ytrain, l, prior):
        if type(prior) is list:
            prior = prior[0]
        w, b = v[0:-1], v[-1]
        p0 = numpy.linalg.norm(w)
        r0 = (l / 2) * numpy.power((p0), 2)
        p11 = numpy.dot(w.T, xtrain[:, ytrain == 1]) + b
        p21 = numpy.multiply(-1, p11)
        p31 = numpy.logaddexp(0, p21)
        r11 = p31.sum(axis=0) * prior / xtrain[:, ytrain == 1].shape[1]

        p12 = numpy.dot(w.T, xtrain[:, ytrain == 0]) + b
        p22 = numpy.multiply(1, p12)
        p32 = numpy.logaddexp(0, p22)
        r12 = p32.sum(axis=0) * (1 - prior) / xtrain[:, ytrain == 0].shape[1]
        return r0 + r11 + r12

    def binary_logRegression(self, xtrain, ytrain, l, quadratic=False, Pc=None):
        if quadratic:
            up = numpy.dot(xtrain.T, xtrain).diagonal().reshape((1, -1))
            xtrain = numpy.vstack((up, xtrain))
        x0 = numpy.zeros(xtrain.shape[0] + 1)
        if not Pc:
            args = (xtrain, ytrain, l)
            x, f, d = fmin_l_bfgs_b(
                self.binary_logreg_obj, x0, args=args, approx_grad=True
            )
        else:
            args = (xtrain, ytrain, l, Pc)
            x, f, d = fmin_l_bfgs_b(
                self.prior_binary_logreg_obj, x0, args=args, approx_grad=True
            )
        wr, br = x[0:-1], x[-1]
        return wr, br

    # Multivariate logistic regression
    def logreg_obj(self, v, xtrain, ytrain, l, k):
        w, b = v[0:-k], v[-k::]
        w = w.reshape((-1, k))
        b = b.reshape((k, -1))
        S = numpy.dot(w.T, xtrain) + b
        Slog = numpy.logaddexp(0, S.T)
        ylog = S - Slog[:, 0]
        Tki = numpy.zeros((k, ytrain.shape[0]))
        for i in range(ytrain.shape[0]):
            Tki[ytrain[i], i] = 1
        TY = numpy.multiply(ylog, Tki)
        p0 = numpy.linalg.norm(w)
        r0 = (l / 2) * numpy.power((p0), 2)
        return r0 - numpy.sum(TY) / xtrain.shape[1]

    def logRegression(self, xtrain, ytrain, l, quadratic=False):
        if quadratic:
            up = numpy.dot(xtrain.T, xtrain).diagonal().reshape((1, -1))
            xtrain = numpy.vstack((up, xtrain))
        k = numpy.unique(ytrain).shape[0]
        x0 = numpy.zeros((xtrain.shape[0] + 1) * k)
        args = (xtrain, ytrain, l, k)
        x, f, d = fmin_l_bfgs_b(self.logreg_obj, x0, args=args, approx_grad=True)
        x = x.reshape((xtrain.shape[0] + 1, k))
        wr, br = x[0:-1, :], x[-1, :]
        br = br.reshape((br.shape[0], 1))
        return wr, br

    # Depreciated
    def hyperparameter_tunning(
        self,
        xtrain,
        ytrain,
        xtest,
        ytest,
        binary=False,
        metric_list=["Error"],
        lambdas=[
            10 ** (-5),
            10 ** (-4),
            10 ** (-3),
            10 ** (-2),
            10 ** (-1),
            0,
            10,
            10 ** (2),
        ],
        models=["linear", "quadratic"],
        save_image=False,
    ):
        """
        Depreciated
        """
        errors = []
        iter = 0
        for j in lambdas:
            iter += 1
            print("iter:", iter, "/", len(lambdas))
            new_models = []
            for i in range(len(models)):
                new_models.append((models[i], j))
            self.train(xtrain, ytrain, models=new_models, binary=binary)
            pred, m = self.evaluate(xtest, ytest, metric_list=metric_list)
            errors.append(m)
        errors = numpy.array(errors).reshape((len(lambdas), len(models), -1))
        if save_image:
            plt.figure()
            for i in range(len(models)):
                plt.plot(lambdas, errors[:, i, :])
            plt.xlabel("L")
            plt.ylabel(metric_list)
            plt.savefig("params_logreg" + ".png")
        return lambdas, errors
