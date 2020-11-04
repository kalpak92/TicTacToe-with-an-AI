import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection as sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

warnings.simplefilter(action='ignore', category=FutureWarning)


class DataLoader:
    @staticmethod
    def preprocess_data_from_csv(dataset_path, split_size):
        print(".. Data Loading ..")

        # data load
        np_arr = np.loadtxt(dataset_path)
        np_x = np_arr[:, :9]
        np_y = np_arr[:, 9]

        np_x_train, np_x_test, np_y_train, np_y_test = \
            Utils.test_train_split(np_x, np_y, split_size)

        return np_x_train, np_x_test, np_y_train, np_y_test

    @staticmethod
    def preprocess_data_from_csv_multi(dataset_path, split_size):
        print(".. Data Loading ..")

        # data load
        np_arr = np.loadtxt(dataset_path)
        np.random.shuffle(np_arr)
        np_x = np_arr[:, :9]
        np_y = np_arr[:, 9:]

        print("ps_np_covariates_X: {0}".format(np_x.shape))
        print("ps_np_treatment_Y: {0}".format(np_y.shape))

        np_x_train, np_x_test, np_y_train, np_y_test = \
            Utils.test_train_split(np_x, np_y, split_size)

        print("np_covariates_X_train: {0}".format(np_x_train.shape))
        print("np_covariates_Y_train: {0}".format(np_y_train.shape))

        print("np_covariates_X_test: {0}".format(np_x_test.shape))
        print("np_covariates_Y_test: {0}".format(np_y_test.shape))

        return np_x_train, np_x_test, np_y_train, np_y_test


class Utils:
    @staticmethod
    def test_train_split(covariates_x, treatment_y, split_size=0.8):
        return sklearn.train_test_split(covariates_x, treatment_y, train_size=split_size)

    @staticmethod
    def get_accuracy_score(y_true, y_pred, normalized=True):
        pred_accu = accuracy_score(y_true, y_pred, normalize=normalized)
        return pred_accu

    @staticmethod
    def plot_knn_accuracy(k_list, knn_score, title):
        plt.title(title)
        plt.plot(k_list, knn_score)
        plt.xlabel("Value of K for KNN")
        plt.ylabel('Validation Accuracy')
        plt.draw()
        plt.savefig("./Plots/" + title, dpi=220)
        plt.clf()


class Regressor:
    def regression_using_knn(self, np_x_train, np_x_test, np_y_train, np_y_test):
        print("Knn Regression")
        param_grid = {"n_neighbors": np.arange(3, 100)}
        knn_gscv = GridSearchCV(KNeighborsRegressor(), param_grid, cv=10)
        knn_gscv.fit(np_x_train, np_y_train)

        best_hyperparams = knn_gscv.best_params_
        optimal_k = best_hyperparams["n_neighbors"]
        print("Optimal: " + str(optimal_k))

        regressor = KNeighborsRegressor(n_neighbors=optimal_k)
        regressor.fit(np_x_train, np_y_train)

        Y_pred = regressor.predict(np_x_test)
        Y_pred = np.where(Y_pred > 0.5, 1, 0)

        total_acc = np.empty(9)
        for i in range(9):
            total_acc[i] = Utils.get_accuracy_score(np_y_test[:, i],
                                                    Y_pred[:, i], normalized=False)

        # print(total_acc)
        # print(np.shape(np_y_test)[0])

        acc = np.sum(total_acc) / (np.shape(np_y_test)[0] * 9)
        print("Accuracy knn: {0}".format(acc))

    def regression_using_mlp(self, np_x_train, np_x_test, np_y_train, np_y_test):
        filename = 'model_params.pkl'
        print(" --->>> MLP Regression")
        # folds = KFold(n_splits=10, shuffle=True, random_state=1)
        param_grid = [
            {
                'max_iter': [1000],
                'hidden_layer_sizes': [(200, 100, 100, 50), (200, 100, 50)],
                'activation': ['tanh', 'relu'],
                'solver': ['adam', 'lbfgs', 'sgd'],
                'alpha': [0.0001],
                'learning_rate': ['constant'],
            }
        ]

        clf = GridSearchCV(MLPRegressor(random_state=1), param_grid, cv=10, n_jobs=-1)
        clf.fit(np_x_train, np_y_train)
        best_score = clf.best_score_
        # print(best_score)
        print("Best parameters set found on development set:")
        print(clf.best_params_)

        best_hyperparams = clf.best_params_
        best_solver = best_hyperparams["solver"]
        best_learning_rate = best_hyperparams["learning_rate"]
        max_iter = best_hyperparams["max_iter"]
        best_layer_size = best_hyperparams["hidden_layer_sizes"]
        best_alpha = best_hyperparams["alpha"]
        best_activation = best_hyperparams["activation"]

        final_clf = MLPRegressor(random_state=20,
                                 max_iter=max_iter, activation=best_activation,
                                 hidden_layer_sizes=best_layer_size,
                                 learning_rate=best_learning_rate,
                                 solver=best_solver,
                                 alpha=best_alpha)

        final_clf.fit(np_x_train, np_y_train)
        y_pred = final_clf.predict(np_x_test)

        y_pred_fixed = np.where(y_pred > 0.5, 1, 0)
        total_acc = np.empty(9)
        for i in range(9):
            total_acc[i] = Utils.get_accuracy_score(np_y_test[:, i],
                                                    y_pred_fixed[:, i], normalized=False)

        acc = np.sum(total_acc) / (np.shape(np_y_test)[0] * 9)
        print("Accuracy MLP: {0}".format(acc))

        pickle.dump(final_clf, open(filename, 'wb'))

    def linear_reg(self, np_x_train, np_x_test, np_y_train, np_y_test):
        print("Linear Regression")
        Y_pred = np.empty((np.shape(np_y_test)[0], np.shape(np_y_test)[1]))
        bias = 1
        for i in range(9):
            y = np_y_train[:, i]
            W = np.linalg.inv(np_x_train.T @ np_x_train) @ np_x_train.T @ y
            W = [weight + bias for weight in W]
            y_pred = np_x_test @ W
            Y_pred[:, i] = y_pred

        Y_pred = (Y_pred == Y_pred.max(axis=1)[:, None]).astype(int)

        total_acc = np.empty(9)
        for i in range(9):
            total_acc[i] = Utils.get_accuracy_score(np_y_test[:, i],
                                                    Y_pred[:, i], normalized=False)

        acc = np.sum(total_acc) / (np.shape(np_y_test)[0] * 9)
        print("Accuracy LR: {0}".format(acc))

    @staticmethod
    def to_labels(Y_pred, t):
        return (Y_pred >= t).astype("int")


def execute_regressor(final_dataset_path, split_size):
    dL = DataLoader()
    np_x_train, np_x_test, np_y_train, np_y_test = \
        dL.preprocess_data_from_csv_multi(final_dataset_path, split_size)

    startRegression = Regressor()
    print("---" * 20)
    print("1. Model: KNN")
    startRegression.regression_using_knn(np_x_train, np_x_test, np_y_train, np_y_test)

    print("---" * 20)
    print("2. Model: MLP")
    startRegression.regression_using_mlp(np_x_train, np_x_test, np_y_train, np_y_test)

    print("---" * 20)
    print("3. Model: Linear Regression")
    startRegression.linear_reg(np_x_train, np_x_test, np_y_train, np_y_test)
