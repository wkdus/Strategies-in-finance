"Lecture 04 index model 2"

import numpy as np                   # linear algebra etc
import matplotlib.pyplot as plt      # plot
import scipy.optimize as optim       # optimization, equation solver
from datetime import datetime as dt  # date, time
import pandas as pd                  # load xlsx, dataframe
import statsmodels.api as sm         # statistical models
import sys                           # system commands, sys.exit('message')


def main():
    # a single index model

    # current risk-free rate (annual). To be used in figures only
    rfCur = 0.02

    # predicted for the future (annual)
    mkt_premium = 0.08  # market risk premium
    mkt_sd = 0.25  # market standard deviation
    alpha_predicted = np.array([0.02, -0.02, 0.01, 0.025, 0])  # abnormal returns
    # 7 securities. The last one is the market portfolio return
    nstocks = alpha_predicted.size

    # load data
    xlsx_load = pd.ExcelFile('lecture04_index_model_data.xlsx')  # xlsx file
    df = xlsx_load.parse(0, index_col=0)  # first worksheet, the first column is the index (row names)
    # >>> df.columns
    # Out[1]:
    # Index(['GE', 'SBUX', 'WMB', 'C', 'qq', 'qq', 'S&P 500',
    #        'Tbill_rate'], dtype='object')

    names_stocks = df.columns[0:nstocks]  # names of stocks
    rm = df['S&P 500']
    rf = df.Tbill_rate/1200
    ex_rstocks_mon_mat = df.iloc[:,0:nstocks] - np.tile(rf.reshape((-1,1)),(1,nstocks))
    ex_rstocks_mon = pd.DataFrame(ex_rstocks_mon_mat,columns=names_stocks)
    ex_rm_mon = pd.Series(rm - rf, name='SP500')  # excess return of market portfolio

    mkt_sd_hist_mon = np.std(ex_rm_mon)  # historical market standard deviation
    var_mon = np.var(ex_rstocks_mon.values,axis=0)
    print('historical market SD (monthly)   : {:8.4f}'.format(mkt_sd_hist_mon))
    print('historical market SD (annualized): {:8.4f}'.format(np.sqrt(12)*mkt_sd_hist_mon))

    # regressions
    betas = np.zeros(nstocks)
    alphas_hist_mon = np.zeros(nstocks)  # monthly historical alpha
    var_res_mon = np.zeros(nstocks)  # monthly residual var
    std_res_mon = np.zeros(nstocks)  # monthly residual SD
    R2 = np.zeros(nstocks)  # R squared

    X = ex_rm_mon.copy()
    X = sm.add_constant(X)
    for i in range(nstocks):
        result = sm.OLS( ex_rstocks_mon[names_stocks[i]], X ).fit()
        betas[i] = result.params['SP500']
        alphas_hist_mon[i] = result.params['const']
        var_res_mon[i] = result.mse_resid
        std_res_mon[i] = var_res_mon[i]**0.5
        R2[i] = result.rsquared

    print(' ** below are monthly returns **')
    print(('               ' + nstocks*' {:>8s}').format(*names_stocks))
    print(('beta           ' + nstocks*' {:8.4f}').format(*betas))
    print(('alpha          ' + nstocks*' {:8.4f}').format(*alphas_hist_mon))
    print(('std of residual' + nstocks*' {:8.4f}').format(*std_res_mon))
    print()
    print(('var total      ' + nstocks*' {:8.4f}').format(*var_mon))
    print(('var due to mkt ' + nstocks*' {:8.4f}').format(*(var_mon - var_res_mon)))
    print(('R squared      ' + nstocks*' {:8.4f}').format(*R2))

    print(' ** above are monthly returns **')

    # draw SML
    expected_returns = rfCur+betas*mkt_premium
    plt.figure()
    plt.scatter(betas, expected_returns, color='red', s=100)
    for i in range(names_stocks.size):
        plt.text(betas[i], expected_returns[i], names_stocks[i],
                 horizontalalignment='right', verticalalignment='bottom',) #,'horizontal','right', 'vertical','bottom')
    plt.plot([0,max(betas)],[rfCur, max(expected_returns)])
    plt.xlabel('beta')
    plt.ylabel('expected return (annualized)')
    plt.title('market risk premium = {:5.3f}'.format(mkt_premium))
    plt.xlim(xmin=0)

    # portfolio analysis begins
    print()
    print(' ** Below are annualized excess returns (or risk premium), unless stated otherwise **')
    print('predicted market risk premium = {:.3f}'.format(mkt_premium))
    print('predicted market sd           = {:.3f}'.format(mkt_sd))

    risk_prem = alpha_predicted + betas*mkt_premium
    var_due_to_mkt = (betas*mkt_sd)**2  # variance due to market
    var_res = var_res_mon*12  # variance due to firm-specific risk  (annualized)
    var_total = var_due_to_mkt + var_res  # total variance
    sd_total = np.sqrt(var_total)  # total volatility (predicted)
    Sr = risk_prem / sd_total  # Sharpe ratio

    print()
    print(('               '+nstocks*' {:>8s}').format(*names_stocks))
    print(('risk premium   '+nstocks*' {:8.4f}').format(*risk_prem))
    print(('var due to mkt '+nstocks*' {:8.4f}').format(*var_due_to_mkt))
    print(('var of residual'+nstocks*' {:8.4f}').format(*var_res))
    print(('var total      '+nstocks*' {:8.4f}').format(*var_total))
    print(('sd total       '+nstocks*' {:8.4f}').format(*sd_total))
    print(('Sharpe ratio   '+nstocks*' {:8.4f}').format(*Sr))

    # compute variance-covariance matrix
    cov_due_to_mkt = np.dot( betas.reshape((-1,1)), betas.reshape((1,-1)) ) *(mkt_sd**2)
    cov_res = np.diag(var_res)  # residuals (firm-specific risks) are assumed to be independent
    cov_total = cov_due_to_mkt + cov_res

    # Compute m-v of random portfolios
    N=1000
    p_mean_rnd = np.zeros(N)
    p_std_rnd = np.zeros(N)
    w_mat = np.random.rand(nstocks,N)
    w_mat_sum = w_mat.sum(axis=0)
    for i in range(N):
        w = w_mat[:,i]/w_mat_sum[i]  # weight vector
        p_mean_rnd[i], p_std_rnd[i] = p_mean_std(w, rfCur+risk_prem, cov_total)

    # Find the minimum variance portfolio
    obj = lambda w: w.dot(cov_total.dot(w))
    w0 = np.ones(nstocks)/nstocks
    bnds = [[-0.05,1]]*nstocks
    cons = {'type': 'eq', 'fun': lambda w:  w.sum()-1}
    sol = optim.minimize(obj,w0, bounds=bnds, constraints=cons)
    if not sol.success:
        sys.exit('cannot find the minimum variance portfolio')  # terminates the code
    min_var_w = sol.x
    minv_mean, minv_std = p_mean_std(min_var_w, rfCur + risk_prem, cov_total)

    # Find the optimal portfolio
    obj = lambda w: -sharpe_ratio(w,risk_prem,cov_total)
    w0 = np.ones(nstocks)/nstocks
    #bnds = [[-0.05,1]]*nstocks
    cons = {'type': 'eq', 'fun': lambda w:  w.sum()-1}
    sol = optim.minimize(obj,w0, bounds=((0,1),(-0.05,1),(-0.05,1),(-0.05,1),(0.8,1)), constraints=cons)
    if not sol.success:
        sys.exit('cannot find the optimal portfolio')  # terminates the code
    opt_w = sol.x ###
    opt_rp, opt_std = p_mean_std(opt_w, risk_prem, cov_total)
    opt_mean = rfCur + opt_rp

    # efficient frontier
    given_rp = np.linspace(min(risk_prem), max(risk_prem), 50)
    p_eff_mean = np.empty_like(given_rp)  # mean of efficient portfolio
    p_eff_std = np.empty_like(given_rp)   # sd of efficient portfolio
    obj = lambda w: w.dot(cov_total.dot(w))
    w0 = np.ones(nstocks)/nstocks
    bnds = [[-0.05,1]]*nstocks
    for i in range(given_rp.size):
        cons = ({'type': 'eq', 'fun': lambda w:  w.sum()-1},
                {'type': 'eq', 'fun': lambda w:  w.dot(risk_prem) - given_rp[i]})
        sol = optim.minimize(obj,w0, bounds=bnds, constraints=cons)
        if not sol.success:
            sys.exit('cannot find the minimum variance portfolio given a risk premium')  # terminates the code
        w1 = sol.x
        p_eff_mean[i], p_eff_std[i] = p_mean_std(w1, rfCur + risk_prem, cov_total)

    # active portfolio and information ratio
    act_w_total = 1 - opt_w[-1]  # weight total to active portfolio (last element is the market)
    act_w = opt_w/act_w_total    # active portfolio weight
    act_w[-1] = 0                # 0 to market
    act_rp, act_std = p_mean_std(act_w, risk_prem, cov_total)
    act_mean = rfCur + act_rp
    act_Sr = act_rp/act_std
    act_alpha = alpha_predicted.dot(act_w)
    act_beta = betas.dot(act_w)
    act_res_var = act_w.dot(cov_res.dot(act_w))
    act_res_sd = np.sqrt(act_res_var)
    info_ratio = act_alpha/act_res_sd
    computed_Sr = np.sqrt(Sr[-1]**2 + info_ratio**2)
    opt_Sr = opt_rp/opt_std

    # a summarizing figure
    plt.figure()
    green = '#00FF00'  # hex strings for green (brighter)
    plt.scatter(p_std_rnd,p_mean_rnd,color=green)  # random portfolio
    plt.plot([0,opt_std*1.5],[rfCur, rfCur+opt_rp*1.5], color='black')              # tangent line
    plt.plot(p_eff_std,p_eff_mean, color='red')  # efficient frontier
    plt.scatter(sd_total, rfCur + risk_prem, color='red', s=100)  # stocks
    for i in range(names_stocks.size):
        plt.text(sd_total[i], rfCur + risk_prem[i], '  ' + names_stocks[i])
    plt.scatter(act_std, act_mean, color='blue', s=100)   # active portf
    plt.text(act_std, act_mean,'  active')
    plt.scatter(minv_std,minv_mean, color='blue', s=100)   # minimum var
    plt.text(minv_std, minv_mean, 'min var  ', horizontalalignment='right')
    plt.scatter(opt_std, opt_mean, color='yellow', edgecolors='black', s=100)
    plt.text(opt_std, opt_mean, 'opt portf  ', horizontalalignment='right')
    plt.xlim(xmin=0)
    plt.xlabel('volatility')
    plt.ylabel('expected return')

    # print results
    print()
    print(('               '+nstocks*' {:>8s}').format(*names_stocks))
    print(('min var portf: '+nstocks*' {:8.4f}').format(*min_var_w))
    print(('alpha predict: '+nstocks*' {:8.4f}').format(*alpha_predicted))
    print(('var of resid : '+nstocks*' {:8.4f}').format(*var_res))
    print(('optimal portf: '+nstocks*' {:8.4f}').format(*opt_w))
    print( 'market weight: {:8.4f}'.format(opt_w[-1])) ##
    print(('active portf:  '+nstocks*' {:8.4f}').format(*act_w))
    print()
    print('active portf risk premimum: {:8.4f}'.format(act_rp))
    print('active portf risk sd      : {:8.4f}'.format(act_std))
    print('active portf risk Sharpe r: {:8.4f}'.format(act_Sr))
    print('active portf alpha        : {:8.4f}'.format(act_alpha))
    print('active portf beta         : {:8.4f}'.format(act_beta))
    print('active portf residual sd  : {:8.4f}'.format(act_res_sd))
    print('active portf info ratio   : {:8.4f}'.format(info_ratio))
    print('optimal portf Sharpe(indirect): {:8.4f}'.format(computed_Sr)) ##
    print('optimal portf Sharpe (direct) : {:8.4f}'.format(opt_Sr)) ##
    print('market portf  Sharpe (direct) : {:8.4f}'.format(Sr[-1]**2))


def sharpe_ratio(w,rp,vcov):
    # returns the Sharpe ratio
    # w: weight vector
    # rp: risk premium (= raw_return - risk_free )
    # vcov: covariance matrix
    (m,s) = p_mean_std(w,rp,vcov)
    return m/s


def p_mean_std(w,ret,vcov):
    # returns (mean, standard deviation) of a portfolio
    # m, s: mean, std
    # w: weight vector
    # ret: expected return
    # vcov: covariance matrix

    m = w.dot(ret)
    s = np.sqrt(w.dot(vcov.dot(w)))
    return (m,s)


main()
plt.show()