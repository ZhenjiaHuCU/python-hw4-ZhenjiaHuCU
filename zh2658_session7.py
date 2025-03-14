import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from rdatasets import data
from scipy.stats import norm

def MyLM(formula, data):
    fit = smf.ols(formula, data=data).fit()
    return fit

from sklearn.datasets import load_iris
data = load_iris()
iris = pd.DataFrame(data=data.data, columns=data.feature_names)
iris.columns = iris.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

myFormula = "sepal_length_cm ~ petal_length_cm"
x = MyLM(myFormula, iris)
print(np.round(x.params[0], 4))


df_cars = data('cars')

def PlotSLRCI(fit, data):
    x = data['speed']
    y = data['dist']

    pred = fit.get_prediction(sm.add_constant(x)).summary_frame()

    plt.scatter(x, y, facecolors='none', edgecolors='black', label='Data')

    plt.plot(x, pred['mean'], color='black', label='Fitted Line')

    plt.plot(x, pred['mean_ci_lower'], linestyle='--', color='grey', label='Lower CI')
    plt.plot(x, pred['mean_ci_upper'], linestyle='--', color='grey', label='Upper CI')

    plt.xlabel('Speed')
    plt.ylabel('Distance')
    plt.legend()
    plt.show()

fit = sm.OLS(df_cars['dist'], sm.add_constant(df_cars['speed'])).fit()
PlotSLRCI(fit, df_cars)

def TMAT1(vec1, vec2):
    states = np.unique(np.concatenate([vec1, vec2]))
    mat = np.zeros((len(states), len(states)))
    for i, s1 in enumerate(states):
        for j, s2 in enumerate(states):
            mat[i, j] = np.sum((vec1 == s1) & (vec2 == s2)) / np.sum(vec1 == s1)
    return mat

rLast = np.repeat(['A', 'B', 'C'], [3, 4, 5])
rNow = np.repeat(['A', 'B', 'C'], [5, 2, 5])
out = TMAT1(rLast, rNow)
answer = np.array([[1,0,0],[0.5,0.5,0],[0,0,1]])
print("Test Passed:", np.array_equal(out, answer))

def Forecast_nPeriod(vec, tmat, n):
    out = np.eye(tmat.shape[0])
    for _ in range(n):
        out = np.dot(out, tmat)

    result = np.dot(vec, out)

    return result

initialStates = np.array([[20, 30, 10]])
tmat = np.array([[1, 0, 0], [0.5, 0.5, 0], [0, 0, 1]])

states1 = Forecast_nPeriod(initialStates, tmat, 1)
states2 = Forecast_nPeriod(initialStates, tmat, 2)
states3 = Forecast_nPeriod(initialStates, tmat, 3)

print("States after 1 period:", states1)
print("States after 2 periods:", states2)
print("States after 3 periods:", states3)

test_result = np.isclose(states3[0][0], 46.25)
print(states3[0][0])
print("Test Passed:", test_result)

def getReturns(pricevec, lag=1):
    returns = (np.array(pricevec[lag:]) - np.array(pricevec[:-lag])) / np.array(pricevec[:-lag])
    return returns

# Example data vector
x = [100, 120, 150, 200]

# Testing the function with various lags
rets1 = getReturns(x, 1)
rets2 = getReturns(x, 2)
rets3 = getReturns(x, 3)

# Test assertions
print("Test 1 Passed:", np.isclose(round(rets1[0], 2), 0.20))
print("Test 1 Passed:", np.isclose(round(rets1[1], 2), 0.25))
print("Test 1 Passed:", np.isclose(round(rets1[2], 2), 0.33))
print("Test 2 Passed:", np.isclose(round(rets2[0], 2), 0.50))
print("Test 3 Passed:", np.isclose(round(rets3[0], 2), 1.00))

def getBondDuration(y, face, couponRate, m, ppy=1):
    c = couponRate * face / ppy
    t = np.arange(1, m * ppy + 1) / ppy
    pvcf = c / (1 + y / ppy)**(t * ppy)
    pvcf[-1] += face / (1 + y / ppy)**(m * ppy)
    duration = np.sum(pvcf * t) / np.sum(pvcf)
    return duration

y = 0.03
face = 2000000
couponRate = 0.04
m = 10
x = getBondDuration(y, face, couponRate, m)
print("Duration:", x, "Expected:", 8.51, "Test Passed:", np.isclose(round(x, 2), 8.51))

x = getBondDuration(y, face, couponRate, m, 1)
print("Duration:", x, "Expected:", 8.51, "Test Passed:", np.isclose(round(x, 2), 8.51))

x = getBondDuration(y, face, couponRate, m, 2)
print("Duration:", x, "Expected:", 8.42, "Test Passed:", np.isclose(round(x, 2), 8.42))

def PercentVaR(r, alpha):
    plt.hist(r, bins=50)
    plt.show()
    out = np.abs(np.quantile(r, 1 - alpha))
    return out

returns = np.random.normal(size=10000)
percentile_value = np.quantile(returns, 0.9772)
print("97.72% Percentile:", percentile_value)

# Unit test
r = np.random.normal(loc=0.05, scale=0.03, size=1000000)
probability2SD = norm.cdf(2)  # 97.72%

myalpha = probability2SD
myPercentVaR = PercentVaR(r, myalpha)
print("Calculated PercentVaR:", myPercentVaR)
print("Test Passed:", np.isclose(round(myPercentVaR, 2), 0.01))


def ES(losses, alpha=None, VaR=None):
    if VaR is not None:
        out = losses[losses > VaR].mean()
    else:
        VaR = np.percentile(losses, alpha * 100)
        out = losses[losses > VaR].mean()
    return out


np.random.seed(42)  # For reproducibility
u = np.random.uniform(0, 100, 100000)

# Test Expected Shortfall calculation with alpha
es_alpha = ES(losses=u, alpha=0.8)
print("Expected Shortfall (alpha=0.8):", round(es_alpha, 0))

# Test Expected Shortfall calculation with given VaR
es_var = ES(losses=u, VaR=80)
print("Expected Shortfall (VaR=80):", round(es_var, 0))

# Output should match these values if the implementation is correct
print("Test Passed (alpha):", np.isclose(round(es_alpha, 0), 90))
print("Test Passed (VaR):", np.isclose(round(es_var, 0), 90))