"Lecture 01_MV_3stocks(필기 및 수정)"

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optim

def main():
    # mean-variance analysis with 3 securities
    
    command = 3
    # 1: draw m-v
    # 2: minimum variance portfolio
    # 3: optimal portfolio
    # 4: minimum variance given a mean
    
    risk_prem = np.array([0.10742, 0.16398, 0.04638, 1.65944])  # risk premium
    vcov = np.array([[0.035669, 0.003501, 0.012257, -0.05201],
                     [0.003501, 0.08769, 0.009573, 0.016151],
                     [0.012257, 0.009573, 0.107058, -0.06531],
                     [-0.05201, 0.016151, -0.06531, 12.54115]])
    # variance-covariance matrix
    # You may compute the estimates of (risk_prem, vcov) using Excel
    
    Sr = risk_prem/np.sqrt(np.diag(vcov))  # Sharpe ratio
    print('Sharpe ratios:', Sr)
    
    # Compute m-v of random portfolios
    N=1000
    p_mean=np.zeros(N)
    p_std=np.zeros(N)
    w_mat = np.random.rand(4,N)
    # 랜덤 숫자를 행이 3개고 n이 0개인 랜덤한 행렬을 뽑은 것이다.
    
    w_mat_sum = w_mat.sum(axis=0)
    # 칼럼별로 다 더해라
    # 포트폴리오에서 임의로 뽑아냈음
    for i in range(N):
        w = w_mat[:,i]/w_mat_sum[i]  # weight vector
        (p_mean[i], p_std[i]) = p_mean_std(w,risk_prem,vcov)
    
    if command in [1, 2, 3, 4]:
        # mean-variance
        # green = '#00FF00'  # hex strings for green
        plt.scatter(p_std,p_mean, s=5, color='green', alpha=0.5)
        plt.scatter(np.sqrt(np.diag(vcov)),risk_prem, color='blue')
        plt.xlim(0,4.0)
        plt.ylim(0,2.0)
        #(ymin=0)
    
    if command==2:
        # Find the minimum variance portfolio
        obj = lambda w: w.dot(vcov.dot(w))
        #목적함수(objective)
        w0 = np.array([1,1,1,1])/4
        cons = {'type': 'eq', 'fun': lambda w:  w.sum()-1}
        # 저 함수가 0이 되는????
        
        min_var_w = optim.minimize(obj,w0, bounds=((0,1),(0,1),(0,1),(0,1)), constraints=cons).x
        print('minimum variance portfolio:', min_var_w)
        (opt_mean, opt_std) = p_mean_std(min_var_w,risk_prem,vcov)
    
        plt.scatter(opt_std,opt_mean, s=100, color='red')
        plt.xlim(0,10.5)

    if command in [3, 4]:
        # Find the optimal portfolio
        obj = lambda w: -sharpe_ratio(w,risk_prem,vcov)
        w0 = np.array([1,1,1,1])/4
        cons = {'type': 'eq', 'fun': lambda w:  w.sum()-1}
        #저 함수가 1이 되게하는 것에서만 움직여라.
        opt_w = optim.minimize(obj,w0, bounds=((0,1),(0,1),(0,1),(0,1)), constraints=cons).x
        print('optimal portfolio:', opt_w)
        (opt_mean, opt_std) = p_mean_std(opt_w,risk_prem,vcov)

        plt.scatter(opt_std,opt_mean, s=100, color='red')
        plt.plot([0,opt_std*1.5],[0,opt_mean*1.5])
        plt.xlim(0.0,4.0)
        plt.ylim(-0.5,2.0)
        
    if command==4:
        # minimum variance given a mean
        N = 21
        given_mean = np.linspace(min(risk_prem),max(risk_prem),N)
        p_mean = np.zeros(N)
        p_std = np.zeros(N)
        
        obj = lambda w: w.dot(vcov.dot(w))
        w0 = np.array([1,1,1,1])/4
        for i in range(N):
            cons = ({'type': 'eq', 'fun': lambda w:  w.sum()-1},
                    {'type': 'eq', 'fun': lambda w:  w.dot(risk_prem) - given_mean[i]})
            w1 = optim.minimize(obj,w0, bounds=((0,1),(0,1),(0,1),(0,1)), constraints=cons).x
            (p_mean[i], p_std[i]) = p_mean_std(w1,risk_prem,vcov)
        #기대수익률이나 위험프리미엄을 여러번 바꿔가면서 하라는 뜻임
        plt.plot(p_std,p_mean, color='black')  # solid, circle, black


def sharpe_ratio(w,ret,vcov):
    # returns the Sharpe ratio
    # w: weight vector
    # ret: risk premium (= raw_return - risk_free )
    # vcov: covariance matrix
    (m,s) = p_mean_std(w,ret,vcov)
    return m/s


def p_mean_std(w,ret,vcov):
    # returns (mean, standard deviation) of a portfolio
    # m, s: mean, std
    # w: weight vector
    # ret: risk premium (= raw_return - risk_free )
    # vcov: covariance matrix
    
    m = w.dot(ret)
    s = np.sqrt(w.dot(vcov.dot(w)))
    return (m,s)


main()
plt.show()