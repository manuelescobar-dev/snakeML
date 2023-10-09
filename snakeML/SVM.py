from matplotlib import pyplot as plt
import numpy
from scipy.optimize import fmin_l_bfgs_b
from snakeML import metrics


class SVM:
    def __init__(self, factr=1e7):
        self.factr = factr
        self.params = []
        self.predictions = []
        self.metrics = []

    def svm_obj(self, alpha, H):
        alpha = alpha.reshape(alpha.size, 1)
        b = numpy.dot(alpha.T, H)
        c = numpy.dot(b, alpha) / 2
        J = (c - alpha.sum())[0, 0]
        gradient = numpy.dot(H, alpha) - 1
        return J, gradient.flatten()

    def primal_solution(self, alpha, D, zi):
        a = numpy.multiply(alpha.reshape((alpha.size, 1)), zi)
        w = numpy.multiply(a, D.T)
        return w.sum(axis=0)

    def extended_data_matrix(self, x, y, k):
        zi = 2 * y - 1
        zi = zi.reshape((zi.size, 1))
        zij = numpy.dot(zi, zi.T)
        newrow = numpy.ones(x.shape[1]) * k
        D = numpy.vstack([x, newrow])
        return D, zij, zi

    def kernel_matrix(self, kernel, x1, x2, args=[]):
        if kernel == "Poly":
            IG = numpy.dot(x1.T, x2) + args[2]
            G = numpy.power(IG, args[3]) + args[0] ** 2
        elif kernel == "RBF":
            a = x1[:, :, numpy.newaxis] - x2[:, numpy.newaxis, :]
            b = numpy.square(numpy.linalg.norm(a, axis=0))
            G = numpy.exp(-args[2] * b) + args[0] ** 2
        return G

    def obj_params(self, xtrain, ytrain, kernel, kernel_args=[]):
        D, zij, zi = self.extended_data_matrix(xtrain, ytrain, kernel_args[0])
        if kernel == "linear":
            G = numpy.dot(D.T, D)
            H = numpy.multiply(zij, G)
        elif kernel == "Poly":
            G = self.kernel_matrix("Poly", xtrain, xtrain, args=kernel_args)
            H = numpy.multiply(zij, G)
        elif kernel == "RBF":
            G = self.kernel_matrix("RBF", xtrain, xtrain, args=kernel_args)
            H = numpy.multiply(zij, G)
        else:
            raise Exception("Kernel not implemented")
        return H, D, zi

    def train(self, xtrain, ytrain, submodels=[("linear", (0, 1))], Pc=None):
        """
        Train SVMs

        Parameters
        ----------
        xtrain : Training data
        ytrain : Training labels
        factr : tolerance
        binary : True if binary classification
        models : list of tuples (kernel, kernel_args)
            - linear: kernel_args=(K,C)
            - Poly: kernel_args=(K,C,c,d)
            - RBF: kernel_args=(K,C,gamma)
            - For example: [("linear", (0,1)), ("Poly", (0,1,2,3))]
        """
        self.params = []
        self.predictions = []
        self.metrics = []
        self.models = submodels
        classes = numpy.unique(ytrain)
        if len(classes) == 2:
            self.isBinary = True
        else:
            self.isBinary = False
        trained = 0
        for i in self.models:
            if type(i) is tuple:
                model = i[0]
                model_args = i[1]
            else:
                print("Enter model arguments")
            x0 = numpy.ones(xtrain.shape[1])
            H, D, zi = self.obj_params(
                xtrain, ytrain, kernel=model, kernel_args=model_args
            )
            args = [H]
            bound = numpy.array([(0, model_args[1]) for i in range(xtrain.shape[1])])
            x, f, d = fmin_l_bfgs_b(
                self.svm_obj, x0, args=args, bounds=bound, factr=self.factr
            )
            if model == "linear":
                w = self.primal_solution(x, D, zi)
                self.params.append(w)
            else:
                a = x * zi.T
                self.params.append([a, xtrain])
            trained += 1
            # print("Model trained:", trained, "/", len(self.models))

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
        metric_list: list of metrics to be evaluated
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
            D, zij, zi = self.extended_data_matrix(xtest, ytest, self.models[i][1][0])
            if self.models[i][0] == "linear":
                score = numpy.dot(self.params[i].T, D)
            elif self.models[i][0] == "Poly" or self.models[i][0] == "RBF":
                k = self.kernel_matrix(
                    self.models[i][0], self.params[i][1], xtest, self.models[i][1]
                )
                score = numpy.dot(self.params[i][0], k)
            if self.isBinary:
                LP = []
                for i in score.flatten():
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
                self.metrics.append(err)
        if return_score:
            return scores
        else:
            return self.predictions, self.metrics

    # Depreciated
    def hyperparameter_tunning(
        self,
        xtrain,
        ytrain,
        xtest,
        ytest,
        binary=False,
        metric_list=["Error"],
        models=["linear", "Poly", "RBF"],
        C=[
            10 ** (-5),
            10 ** (-4),
            10 ** (-3),
            10 ** (-2),
            10 ** (-1),
            1,
            10 ** (1),
            10 ** (2),
        ],
        save_image=False,
    ):
        """
        Depreciated
        """
        errors = []
        iter = 0
        for j in C:
            iter += 1
            print("iter:", iter, "/", len(C))
            new_models = []
            for i in range(len(models)):
                if type(models[i]) is tuple:
                    model = models[i][0]
                    model_args = [models[i][1][0]] + [j] + models[i][1][1:]
                    new_models.append((model, model_args))
            self.train(xtrain, ytrain, models=new_models, binary=binary)
            pred, m = self.evaluate(xtest, ytest, metric_list=metric_list)
            errors.append(m)
        errors = numpy.array(errors).reshape((len(C), len(models), -1))
        if save_image:
            plt.figure()
            for i in range(len(models)):
                plt.plot(C, errors[:, i, :])
            plt.xlabel("L")
            plt.ylabel(metric_list)
            plt.savefig("params_logreg" + ".png")
        return C, errors
