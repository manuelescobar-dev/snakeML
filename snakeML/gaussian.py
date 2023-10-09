from matplotlib import pyplot as plt
import numpy
import math
from snakeML.numpy_transformations import mean_cov, wc_cov, mean
from snakeML import density_estimation, numpy_transformations, metrics


class generativeClassifier:
    def __init__(self):
        self.params = []
        self.predictions = []
        self.metrics = []

    # Model training
    def train(
        self,
        x_train,
        y_train,
        submodels=["logMVG", "logNBG", "logTiedMVG", "logTiedNBG"],
        Pc=None,
    ):
        """
        Trains gaussian models

        Parameters
        ----------
        xtrain: training data matrix
        ytrain: training labels
        submodels: list of submodels to be trained
            - 'logMVG'
            - 'logNBG'
            - 'logTiedMVG'
            - 'logTiedNBG'
            - ('GMM', [alpha, k, psi])
            - ('tiedGMM', [alpha, k, psi])
            - ('diagGMM', [alpha, k, psi])
        Pc: list of priors for each submodel
        """
        self.params = []
        self.predictions = []
        self.metrics = []
        self.models = submodels
        if not Pc:
            classes = numpy.unique(y_train)
            self.Pc = 1 / classes.size
        for modelf in self.models:
            if type(modelf) is tuple:
                model = modelf[0]
                args = modelf[1]
            else:
                model = modelf
            match (model):
                case ("MVG"):
                    isTied = False
                    mus, covs = self.MVG(x_train, y_train, isTied)
                    self.params.append((mus, covs))
                case ("logMVG"):
                    isTied = False
                    mus, covs = self.MVG(x_train, y_train, isTied)
                    self.params.append((mus, covs))
                case ("TiedMVG"):
                    isTied = True
                    mus, covs = self.MVG(x_train, y_train, isTied)
                    self.params.append((mus, covs))
                case ("logTiedMVG"):
                    isTied = True
                    mus, covs = self.MVG(x_train, y_train, isTied)
                    self.params.append((mus, covs))
                case ("NBG"):
                    isTied = False
                    mus, covs = self.NBG(x_train, y_train, isTied)
                    self.params.append((mus, covs))
                case ("logNBG"):
                    isTied = False
                    mus, covs = self.NBG(x_train, y_train, isTied)
                    self.params.append((mus, covs))
                case ("TiedNBG"):
                    isTied = True
                    mus, covs = self.NBG(x_train, y_train, isTied)
                    self.params.append((mus, covs))
                case ("logTiedNBG"):
                    isTied = True
                    mus, covs = self.NBG(x_train, y_train, isTied)
                    self.params.append((mus, covs))
                case ("GMM"):
                    isTied = True
                    if len(args) == 2 and type(args[1]) is int:
                        mus, covs, weights = self.LBGclasses(x_train, y_train, k=args)
                    elif len(args) == 2 and type(args[1]) is list:
                        mus, covs, weights = self.LBGclasses(
                            x_train, y_train, alpha=args[0], k=args[1]
                        )
                    elif len(args) == 3:
                        mus, covs, weights = self.LBGclasses(
                            x_train, y_train, alpha=args[0], k=args[1], psi=args[2]
                        )
                    self.params.append((mus, covs, weights))
                case ("tiedGMM"):
                    isTied = True
                    if len(args) == 2 and type(args[1]) is int:
                        mus, covs, weights = self.LBGclasses(
                            x_train, y_train, k=args, tied=True
                        )
                    elif len(args) == 2 and type(args[1]) is list:
                        mus, covs, weights = self.LBGclasses(
                            x_train, y_train, alpha=args[0], k=args[1], tied=True
                        )
                    elif len(args) == 3:
                        mus, covs, weights = self.LBGclasses(
                            x_train,
                            y_train,
                            alpha=args[0],
                            k=args[1],
                            psi=args[2],
                            tied=True,
                        )
                    self.params.append((mus, covs, weights))
                case ("diagGMM"):
                    isTied = True
                    if len(args) == 2 and type(args[1]) is int:
                        mus, covs, weights = self.LBGclasses(
                            x_train, y_train, k=args, diag=True
                        )
                    elif len(args) == 2 and type(args[1]) is list:
                        mus, covs, weights = self.LBGclasses(
                            x_train, y_train, alpha=args[0], k=args[1], diag=True
                        )
                    elif len(args) == 3:
                        mus, covs, weights = self.LBGclasses(
                            x_train,
                            y_train,
                            alpha=args[0],
                            k=args[1],
                            psi=args[2],
                            diag=True,
                        )
                    self.params.append((mus, covs, weights))

    # Model evaluation
    def evaluate(self, x_test, y_test, metrics_list=["Accuracy"], return_score=False):
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
        for i in range(len(self.models)):
            if len(self.models[i]) == 2:
                model = self.models[i][0]
                args = self.models[i][1]
            else:
                model = self.models[i]
                args = []
            if model == "GMM" or model == "tiedGMM" or model == "diagGMM":
                # FOR GMMs
                score = []
                for h in range(len(self.params[i][0])):
                    ll = self.ll_gaussian(
                        x_test, self.params[i][0][h], self.params[i][1][h]
                    )
                    Sj = density_estimation.SJoint(ll, pi=self.params[i][2][h])
                    Marginal = density_estimation.SMarginal(Sj)
                    score.append(Marginal)
                score = numpy.array(score)
                score = score.reshape((len(self.params[i][0]), -1))
                pred = numpy.argmax(score, axis=0)
            else:
                # For simple gaussian models
                if i == "MVG" or i == "NBG" or i == "TiedMVG" or i == "TiedNBG":
                    log = False
                else:
                    log = True
                ll = self.ll_gaussian(
                    x_test, self.params[i][0], self.params[i][1], log=log
                )
                score = density_estimation.SPost_from_ll(ll, self.Pc, log=log)
                pred = numpy.argmax(score, axis=0)
            if return_score:
                scores.append(score)
            else:
                err = metrics.test_metrics(pred, y_test, metrics_list, args=score)
                self.predictions.append(pred)
                self.metrics.append(err)
        if return_score:
            return scores
        else:
            return self.predictions, self.metrics

    # SIMPLE GAUSSIAN MODELS METHODS
    # Log-likelihood matrix for simple gaussian distributions
    def ll_gaussian(self, xtest, mu, C, log=True):
        score = []
        for i in range(len(C)):
            C_inv = numpy.linalg.inv(C[i])
            det = numpy.linalg.slogdet(C[i])[1]
            M = xtest.shape[0]
            log_pi = math.log(2 * math.pi)
            x_mu = numpy.subtract(xtest, mu[i].reshape((-1, 1)))
            r1 = numpy.dot(x_mu.T, C_inv)
            r2 = numpy.diagonal(numpy.dot(r1, x_mu))
            result = (-M * log_pi - det - r2) / 2
            if log:
                score.append(result)
            else:
                score.append(numpy.exp(result))
        return numpy.array(score)

    def MVG(self, x_train, y_train, tied=False):
        mus = []
        covs = []
        if tied:
            C = wc_cov(x_train, y_train)
        for i in numpy.unique(y_train):
            if tied:
                mu = mean(x_train[:, y_train == i])
            else:
                mu, C = mean_cov(x_train[:, y_train == i])
            mus.append(mu)
            covs.append(C)
        return mus, covs

    def NBG(self, x_train, y_train, tied=False):
        mus = []
        covs = []
        if tied:
            C = wc_cov(x_train, y_train)
            C = C * numpy.identity(C.shape[0])
        for i in numpy.unique(y_train):
            if tied:
                mu = mean(x_train[:, y_train == i])
            else:
                mu, C = mean_cov(x_train[:, y_train == i])
                C = C * numpy.identity(C.shape[0])
            mus.append(mu)
            covs.append(C)
        return mus, covs

    # GAUSSIAN MIXTURE MODELS METHODS
    # EM algorithm for gaussian mixture models
    def EM(
        self,
        D,
        mus,
        covs,
        weights,
        threshold=10 ** (-6),
        psi=None,
        diagonal=False,
        tied=False,
    ):
        l_old = -numpy.inf
        lg = numpy.inf
        g = len(mus)
        mu_1 = numpy.array(mus)
        cov_1 = numpy.array(covs)
        w_1 = numpy.array(weights)
        ll = self.ll_gaussian(D, mu_1, cov_1)
        Sj = density_estimation.SJoint(ll, pi=w_1)
        Marginal = density_estimation.SMarginal(Sj)
        r = density_estimation.SPost(Sj, Marginal, exp=True)
        Zg = r.sum(axis=1).reshape((-1, 1))
        # print(cov_1.shape)
        if diagonal:
            covnew = []
            for k in range(g):
                Sigma_g = cov_1[k, :, :] * numpy.eye(cov_1.shape[1])
                covnew.append(Sigma_g)
            cov_1 = covnew
        if tied:
            Sigma_g = numpy.zeros((g, cov_1.shape[1], cov_1.shape[1]))
            for k in range(g):
                Sigma_g += Zg[k, :] * cov_1[k, :, :]
            cov_1 = Sigma_g / D.shape[1]
        if psi:
            covnew = []
            for k in range(g):
                cov_1 = numpy.array(cov_1)
                U, s, _ = numpy.linalg.svd(cov_1[k, :, :])
                s[s < psi] = psi
                covnew.append(numpy.dot(U, numpy_transformations.mcol(s) * U.T))
            cov_1 = covnew
        while lg >= threshold:
            ll = self.ll_gaussian(D, mu_1, cov_1)
            Sj = density_estimation.SJoint(ll, pi=w_1)
            Marginal = density_estimation.SMarginal(Sj)
            r = density_estimation.SPost(Sj, Marginal, exp=True)
            Fg = numpy.dot(r, D.T)
            Zg = r.sum(axis=1).reshape((-1, 1))
            mu_1 = Fg / Zg
            w_1 = Zg
            w_1 = (w_1 / w_1.sum()).reshape((-1, 1))
            Sg = []
            cov_1 = []
            b = []
            for i in range(g):
                psg = numpy.zeros((D.shape[0], D.shape[0]))
                for j in range(D.shape[1]):
                    y = r[i, j]
                    xi = D[:, j].reshape((-1, 1))
                    xii = numpy.dot(xi, xi.T)
                    psg += y * xii
                Sg.append(psg)
                b.append(
                    numpy.dot(mu_1[i, :].reshape((-1, 1)), mu_1[i, :].reshape((1, -1)))
                )
            Sg = numpy.array(Sg)
            a = Sg / Zg.reshape((-1, 1, 1))
            b = numpy.array(b)
            cov_1 = a - b
            if diagonal:
                covnew = []
                for k in range(g):
                    Sigma_g = cov_1[k, :, :] * numpy.eye(cov_1.shape[1])
                    covnew.append(Sigma_g)
                cov_1 = covnew
            if tied:
                Sigma_g = numpy.zeros((g, cov_1.shape[1], cov_1.shape[1]))
                for k in range(g):
                    Sigma_g += Zg[k, :] * cov_1[k, :, :]
                cov_1 = Sigma_g / D.shape[1]
            if psi:
                covnew = []
                for k in range(g):
                    cov_1 = numpy.array(cov_1)
                    U, s, _ = numpy.linalg.svd(cov_1[k, :, :])
                    s[s < psi] = psi
                    covnew.append(numpy.dot(U, numpy_transformations.mcol(s) * U.T))
                cov_1 = covnew
            l = Marginal.mean()
            lg = l - l_old
            l_old = l
            # print("loss:",lg)
        return mu_1, cov_1, w_1

    def LBG(self, D, alpha=0.1, num_splits=2, psi=None, diag=False, tied=False):
        mu, C = numpy_transformations.mean_cov(D)
        mu = numpy.array([mu])
        C = numpy.array([C])
        w = numpy.array([1]).reshape((-1, 1, 1))
        for j in range(num_splits):
            mu_split = []
            C_split = []
            w_split = []
            for i in range(len(mu)):
                U, s, Vh = numpy.linalg.svd(C[i])
                d = U[:, 0:1] * s[0] ** 0.5 * alpha
                mu_split.append(mu[i, :, :] - d)
                mu_split.append(mu[i, :, :] + d)
                C_split.append(C[i, :, :])
                C_split.append(C[i, :, :])
                w_split.append(w[i, :, :] / 2)
                w_split.append(w[i, :, :] / 2)
            mu, C, w = self.EM(
                D, mu_split, C_split, w_split, psi=psi, tied=tied, diagonal=diag
            )
            mu = numpy.array(mu).reshape((-1, D.shape[0], 1))
            C = numpy.array(C)
            w = numpy.array(w).reshape((-1, 1, 1))
            # print(mu.shape, C.shape, w.shape)
        return mu, C, w

    def LBGclasses(
        self, xtrain, ytrain, alpha=0.1, k=[2, 2], psi=0.01, diag=False, tied=False
    ):
        mus = []
        covs = []
        weights = []
        splits_tar = int(math.log2(k[1]))
        splits_ntar = int(math.log2(k[0]))
        k = [splits_ntar, splits_tar]
        for i in numpy.unique(ytrain):
            mu, C, w = self.LBG(xtrain[:, ytrain == i], alpha, k[i], psi, diag, tied)
            mus.append(mu)
            covs.append(C)
            weights.append(w)
        return mus, covs, weights

    def hyperparameter_tunning(
        self,
        xtrain,
        ytrain,
        xtest,
        ytest,
        metric=["Error"],
        bounds=6,
        models=["GMM", "tiedGMM", "diagGMM"],
        k_args=[],
        save_image=False,
    ):
        """
        * Only for Gaussian Mixture Models and binary classification tasks
        default metric: error
        """
        components = numpy.arange(bounds)
        errors = []
        iter = 0
        for j in components:
            for jm in components:
                iter += 1
                new_models = []
                for i in range(len(models)):
                    model_args = models[i][1]
                    args = [model_args[0]]
                    args.append([j, jm])
                    args.extend(model_args[1:])
                    new_models.append((models[i][0], args))
                print(
                    "iter:",
                    iter,
                    "/",
                    len(components) * len(components),
                    " | ",
                    new_models,
                )
                self.train(xtrain, ytrain, models=new_models)
                pred, m = self.evaluate(xtest, ytest, metric=metric)
                errors.append(m)
        errors = numpy.array(errors)
        if save_image:
            plt.figure()
            for i in range(len(models)):
                plt.plot(components, errors[:, i, :])
            plt.xlabel("C")
            plt.ylabel(metric)
            plt.savefig("params_GMM" + ".png")
