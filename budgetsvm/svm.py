import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import accuracy_score

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from budgetsvm import kernel
from budgetsvm import optimization as opt


class SVC(ClassifierMixin, BaseEstimator):
    def __init__(self, C=1, kernel=kernel.GaussianKernel(), budget=None):
        self.C = C
        self.kernel = kernel
        self.budget = budget

    def __repr__(self):
        repr = "SVM("
        if self.C != 1:
            repr += f"C={self.C}, "

        if self.kernel != kernel.GaussianKernel():
            repr += f"kernel={self.kernel}, "

        if self.budget is not None:
            repr += f"budget={self.budget}, "

        if repr[-2:] == ", ":
            repr = repr[:-2]

        return repr + ")"

    def __encode_label(self, original):
        # map [0,1] labels to [-1,1] labels
        return -1 if original == 0 else 1

    def __vec_encode_label(self, it):
        return np.vectorize(self.__encode_label)(it)

    def __decode_label(self, encoded):
        # map [-1,1] labels to original labels
        return self.classes_[0] if encoded == -1 else self.classes_[1]

    def __vec_decode_label(self, it):
        return np.vectorize(self.__decode_label)(it)

    def _more_tags(self):
        """ From sklearn documentation https://scikit-learn.org/stable/developers/develop.html

        pairwise (default=False)
            This boolean attribute indicates whether the data (X) fit and similar methods consists of pairwise measures
            over samples rather than a feature representation for each sample. It is usually True where an estimator has
            a metric or affinity or kernel parameter with value ‘precomputed’. Its primary purpose is to support a
            meta-estimator or a cross validation procedure that extracts a sub-sample of data intended for a pairwise
            estimator, where the data needs to be indexed on both axes. Specifically, this tag is used by _safe_split
            to slice rows and columns.
        """
        return {"pairwise": self.kernel.precomputed}

    def fit(self, X, y, warn=False, solver=opt.GurobiSolver(), keep_alpha_equal_C=False):
        X, y = check_X_y(X, y)

        if self.kernel.precomputed and X.shape[0] != X.shape[1]:
            raise ValueError("Precomputed kernel matrix must be square.")

        self.classes_, y = np.unique(y, return_inverse=True)
        if len(self.classes_) == 1:
            raise ValueError("Classifier can't train when only one class is present.")
        if len(self.classes_) > 2:
            raise ValueError(
                "Classifier can't train when more than two classes are present."
            )

        # store len of training set when using precomputed kernel. predict() needs it to validate input size
        if self.kernel.precomputed:
            self.train_set_size_ = len(X)

        y = self.__vec_encode_label(y)

        alpha, optimal, mip_gap = solver.solve(
            X, y, C=self.C, kernel=self.kernel, budget=self.budget
        )

        self.obj_ = solver.obj_val
        self.mip_gap = mip_gap

        if keep_alpha_equal_C:
            sv_mask = (0 < alpha) & (alpha <= self.C)
        else:
            sv_mask = (0 < alpha) & (alpha < self.C)

        if self.kernel.precomputed:
            self.sv_mask_ = sv_mask

        self.alpha_ = alpha[sv_mask]

        # keep track to see how many alpha==C
        self.alpha_eq_c_ = alpha[alpha == self.C]
        self.alpha_lt_c_ = alpha[alpha < self.C]

        if not self.kernel.precomputed:
            self.X_ = X[sv_mask]
        self.y_ = y[sv_mask]

        if not self.kernel.precomputed:
            bs = [
                y_i - self.__dotprod(x)
                for x, y_i, a in zip(self.X_, self.y_, self.alpha_)
            ]
        else:
            bs = [
                y_i - self.__dotprod(x)
                for x, y_i, a in zip(X[self.sv_mask_], self.y_, self.alpha_)
            ]

        if not bs:
            raise FitFailedWarning("no SV founds")

        self.b_ = np.mean(bs)
        if warn and np.std(bs) > 1e-4:
            print("warning: computed values for b are", bs)

        self.optimal_ = optimal
        return self

    def __dotprod(self, x_new):
        """
        x_new is an unknown predict sample if kernel is not precomputed
        x_new is an array of size n_training_sample where pos i has the kernel value for the predict sample and the i-st
            training sample
        """
        if self.kernel.precomputed:
            return np.sum(
                [
                    a * y_i * precomputed_kernel_value
                    for precomputed_kernel_value, y_i, a in zip(
                        x_new[self.sv_mask_], self.y_, self.alpha_
                    )
                ]
            )

        return np.sum(
            [
                a * y_i * self.kernel.compute(x, x_new)
                for x, y_i, a in zip(self.X_, self.y_, self.alpha_)
            ]
        )

    def __decision_function(self, X):
        """
        X [samples] if not precomp else
        X n_samples_test, n_samples_train matrix
        """
        return np.array([self.__dotprod(x) + self.b_ for x in X])

    def predict(self, X):
        """Predict class on samples in X.

        :param X : array-like of shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).
        :returns ndarray of shape (n_samples,) the predicted values.
        """
        check_is_fitted(self)
        if self.kernel.precomputed:
            if X.shape[1] != self.train_set_size_:
                raise ValueError("predict when model has been trained with precomputed kernel expects X of shape "
                                 "(n_sample_test,n_samples_train)")
        else:
            X = check_array(X)
        encoded_label = np.sign(self.__decision_function(X))
        return self.__vec_decode_label(encoded_label)

    def score(self, X, y, **kwargs):
        return accuracy_score(self.predict(X), y)


if __name__ == "__main__":
    # print('lanciato script')
    # from sklearn.model_selection import GridSearchCV

    # c_values = [0.1, 1, 10]
    # kernel_values = [kernel.GaussianKernel(0.1), kernel.GaussianKernel(3)]
    # grid = {'C': c_values, 'kernel': kernel_values}

    # X = np.random.uniform(low=0, high=1, size=(50, 2))
    # labeling_funct = lambda x: 1 if np.dot([-1, 1], x) > 0 else -1
    # y = np.array(list(map(labeling_funct, X)))

    # print('classic SVM model')
    # gs = GridSearchCV(SVC(), grid, cv=2)
    # gs.fit(X, y)
    # model = gs.best_estimator_
    # print(model)

    # print('budgeted SVM model')
    # gs = GridSearchCV(SVC(budget=40), grid, cv=2)
    # gs.fit(X, y)
    # model = gs.best_estimator_
    # print(model)
    # print(sum(model.alpha > 0))

    from kernel import LinearKernel

    def generate_data(n=10, split=0.5):
        y = np.random.uniform(size=n)
        n_pos = int(n * split)
        y_pos = y[:n_pos]
        y_neg = y[n_pos:]
        return np.array([[0.2, y] for y in y_pos] + [[0.8, y] for y in y_neg]), [
            1
        ] * n_pos + [-1] * (n - n_pos)

    X, y = generate_data(n=100)

    svc = SVC(C=10, kernel=LinearKernel(), budget=None)
    svc.fit(X, y)
    print(f"classic SVM used {len(svc.alpha_)} SVs.")

    svc = SVC(C=10, kernel=LinearKernel(), budget=3)
    svc.fit(X, y)
    print(f"budget SVM used {len(svc.alpha_)} SVs.")
