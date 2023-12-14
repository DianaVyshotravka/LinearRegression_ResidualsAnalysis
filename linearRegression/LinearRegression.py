from collections import namedtuple
from math import sqrt
from typing import Any
import numpy as np
from scipy.stats import f


class LinearRegression:
    """
    Performs regresional analysis for given dataset
    """

    def __init__(self):
        """Creates a linearRegression class object
        """
        self.error: float = None
        self.slopes: np.ndarray = None
        self.intercept: float = None
        self.MSE: float = None
        self.RMSE: float = None
        self.MAE: float = None

    def predict(self, x: Any):
        """Predict data from given values
        Args:
            data:
        """
        return (x * self.slopes).sum(axis=1) + self.intercept + self.error

    def evaluate(self, Y: Any, X: Any) -> namedtuple:
        """
        Calculates MSE, RMSE and MAE
        Args:
            Y: dependent variable
            X: independent variable

        Returns:
            named tuple that contains evaluating results
        """
        residuals = Y - self.predict(X)
        self.MSE = ((residuals ** 2).sum()) / len(Y)
        self.RMSE = sqrt(self.MSE)
        self.MAE = abs(residuals).sum() / len(Y)

        return namedtuple('RegressionEvaluation', ('MSE', 'RMSE', 'MAE'))(self.MSE, self.RMSE, self.MAE)

    def fit(self, Y: Any, X: Any) -> object:
        """
        Fits the model
        Args:
            Y: dependent variable data for training
            X: independent variable data for training

        Returns: nothing
        """
        X = np.array(X)
        Y = np.array(Y)

        X = np.concatenate((np.full((X.shape[0], 1), 1), X), axis=1)
        b = np.linalg.inv(X.T @ X) @ X.T @ Y
        self.intercept = b[0]
        self.slopes = b[1:]
        self.error = self.__calculate_error(Y, np.delete(X, 0, axis=1))
        pass

    def __calculate_error(self, y: np.ndarray, x: np.ndarray):
        return (y - ((x * self.slopes).sum(axis=1) + self.intercept)).mean()


def compare(a: LinearRegression, b: LinearRegression, y: np.ndarray, x: np.ndarray, colid: int | list) -> namedtuple:
    """
    Performs a statistical test to depend differency in accuracy of two linear models with null hypotesis that additional parameters aren't statisticly significant, and alternative hypotesis that additional parameters are statisticly significant
    Args:
        a: model that be compared
        b: second regression model to be compared
        y: evaluating dataset predicted value
        x: evaluating dataset independent values
        colid: id of column(s) that will be excluded from x

    Returns: namedtuple
    """

    testing_result = namedtuple('TestingResult', ['statistic', 'pvalue'])
    a_residuals = y - a.predict(x)
    b_residuals = y - b.predict(np.delete(x, colid, axis=1))
    #
    # F = ((b_residuals ** 2).sum() - ((a_residuals ** 2).sum()
    #                                  ) / (len(b.slopes)) - len(a.slopes) / (
    #             ((a_residuals ** 2).sum()) / (len(y) - len(a.slopes)))

    F = ((b_residuals ** 2).sum() - ((a_residuals ** 2).sum()
                                      ) / (len(a.slopes)) - len(b.slopes)) / (
                 ((a_residuals ** 2).sum()) / (len(y) - len(a.slopes)))

    return testing_result(F, f.sf(F, 1, len(a.slopes) - len(b.slopes)))
