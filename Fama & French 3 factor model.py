"Lecture 05_Fama French 3 factor model"

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optim
import pandas as pd
import statsmodels.api as sm


def main():
    # Fama-French 3 factor model

    # load data
    df = pd.ExcelFile('lecture05_FF3F_data.xlsx').parse(0)  # first sheet
    # >>> df.columns.tolist()
    # ['Date', 'Fidelity Magellan Fund', 'Mkt-RF', 'SMB', 'HML', 'Mom', 'RF']

    Y = pd.Series((df['Fidelity Magellan Fund'] - df.RF)/100, name='fund_rf')
    X = df[['Mkt-RF','SMB','HML']]/100
    X = sm.add_constant(X)

    # run a regression
    result = sm.OLS( Y, X ).fit()
    print(result.summary())
    print('')
    print(result.params)


main()
# plt.show()