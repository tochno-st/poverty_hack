import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import norm
from scipy.integrate import quad
import math
import numpy as np

class HackParams:
    def __init__(self, average_income, poverty_line, poverty):
        """Initialize class with parameters from Rosstat."""
        self.average_income = average_income
        self.poverty_line = poverty_line
        self.poverty = poverty/100

    def find_upper_limit(self):
        upper_limit, = fsolve(self.equation, 0, self.poverty)
        return upper_limit

    def find_params(self):
        """Find mean and standard deviation of lognormal distribution used by Rosstat."""
        u_limit = self.find_upper_limit()
        stdev = u_limit + math.sqrt(u_limit**2+2*math.log(self.average_income/self.poverty_line))
        mu = math.log(self.average_income) - 0.5*stdev**2
        return mu, stdev

    def count_poverty(self, new_line):
        """Count poverty based on some value of poverty line."""
        mu, sigma = self.find_params()
        ln_x0 = math.log(self.average_income) - 0.5 * sigma ** 2
        u = (math.log(new_line) - ln_x0) / sigma
        integral, error = quad(self.integrand, -np.inf, u)
        result = round((1 / math.sqrt(2 * math.pi)) * integral, 4)
        return result*100

    @staticmethod
    def equation(x, value):
        return norm.cdf(x) - value

    @staticmethod
    def integrand(x):
        return np.exp((-x ** 2) / 2)