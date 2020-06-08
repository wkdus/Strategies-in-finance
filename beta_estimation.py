"Lecture 03 beta_estimation(새거)"

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optim
import pandas as pd
import statsmodels.api as sm

def main():
    # Compute beta and alpha of a stock

    # load data
    xlsx_load = pd.ExcelFile('lecture03_beta_estimation_data.xlsx')  # xlsx file
    df = xlsx_load.parse(0)  # first worksheet
    # >>> df.columns
    # Out[1]: Index(['DATE', 'HP', 'S&P 500', 'Tbill_rate'], dtype='object')

    r_m = df['S&P 500']
    r_hp = df.HP
    r_f = df.Tbill_rate/1200   # convert to decimal, per month

    # run a regression
    Y = r_hp - r_f
    Y.name = 'HP'
    X = pd.Series(r_m - r_f, name='SP500')
    X = sm.add_constant(X)
    result = sm.OLS( Y, X ).fit()
    print(result.summary())
    print('')
    print(result.params)

    print('')
    a = result.params['const']
    b = result.params['SP500']
    print('alpha = ', a)
    print('beta  = ', b)

    plt.scatter(X.SP500,Y)
    x1 = min(X.SP500)
    x2 = max(X.SP500)
    plt.plot([x1, x2], a+b*[x1,x2])


main()
plt.show()