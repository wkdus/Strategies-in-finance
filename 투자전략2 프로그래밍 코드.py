# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:41:41 2018

@author: Hyunjeong Kang
"""
import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.optimize as optim

#################################### file #####################################
os.chdir(r"C:\Users\Hyunjeong Kang\Desktop")
file = pd.ExcelFile(r'Data.xlsx')
############################# 1. 투자 매력도 산출 ##############################
##### 1) 벨류에이션
Data2 = file.parse('1-2&3')
Data2.index = Data2.pop('Name')
Rim = file.parse('RIM')
Rim.index = Rim.pop('Name')
##### V: Size - 10%
mv = Data2.iloc[:,0]
mv.columns = 'mv'
##### E/P:Per의 역수(시가총액합/12개월 예상 순이익합) - 25%
ep = 1/Data2.iloc[:,-2]
ep.columns = 'ep'
##### V/P: RIM model - 15%
Capm = Data2.iloc[:,2]*Data2.iloc[:,3]+Data2.iloc[:,2]
Dividend = Rim.iloc[:,0]
roe = Rim.iloc[:,3]
Bookvalue = Rim.iloc[:,1]-Rim.iloc[:,2]

Capm = pd.DataFrame(Capm)
Capm.columns=['rim']
Dividend = pd.DataFrame(Dividend)
Dividend.columns=['rim']
roe = pd.DataFrame(roe)
roe.columns=['rim']
Bookvalue = pd.DataFrame(Bookvalue)
Bookvalue.columns=['rim']


g= (1-Dividend)*roe
Rim = Bookvalue+((roe-Capm)*Bookvalue)/(Capm-g)


################################ 2) 모멘텀  ###################################
################################ EPS 수정비율 #################################
EPSmodifiedRatio = file.parse('2-1')
EPSmodifiedRatio.index = EPSmodifiedRatio.pop('Name')
epsratio = EPSmodifiedRatio.iloc[:,0]
epsratio = pd.DataFrame(epsratio)
epsratio.columns = ['eps']
############################ 12분기 순이익 모멘텀 ##############################
NetProfitMomentum = file.parse('2-2')
NetProfitMomentum.index = NetProfitMomentum.pop('date')

A = dict()
for i in range(len(NetProfitMomentum.T)):
    All_Data = pd.DataFrame(NetProfitMomentum[NetProfitMomentum.columns[i]])
    All_Data.columns = ['Y']
    All_Data = All_Data.dropna()
    if len(All_Data)>1:
        reg = smf.ols('Y~1',data=All_Data).fit(cov_type='HAC',cov_kwds={'maxlags':1})
        A[NetProfitMomentum.columns[i]] = reg.tvalues[0]
    else:
        A[NetProfitMomentum.columns[i]] = np.NAN

npm = pd.DataFrame(pd.Series(A))
npm.columns = ['npm']
Mean = npm.mean()
npm = npm.fillna(Mean)
############################## 120일 주가 모멘텀 ##############################
StockMomentum120 = file.parse('2-3')
StockMomentum120.index = StockMomentum120.pop('date')

A = dict()
for i in range(len(StockMomentum120.T)):
    All_Data = pd.DataFrame(StockMomentum120[StockMomentum120.columns[i]])
    All_Data.columns = ['Y']
    reg = smf.ols('Y~1',data=All_Data).fit(cov_type='HAC',cov_kwds={'maxlags':120})
    A[StockMomentum120.columns[i]] = reg.tvalues[0]

sm120 = pd.DataFrame(pd.Series(A))
sm120.columns = ['sm120']
##############################################################################
Attractive_Investment = pd.concat([epsratio*0.25,npm*0.125,sm120*0.125,Rim*0.15,ep*0.25,mv*0.10],axis=1)
Attractive_Investment = Attractive_Investment.cumsum(axis=1).iloc[:,-1]
Attractive_Investment = pd.DataFrame(Attractive_Investment)
Attractive_Investment = Attractive_Investment.sort_values(by='시가총액')
Attractive_Investment['Rank'] = 1
Attractive_Investment['Rank'] = Attractive_Investment['Rank'].cumsum(axis=0)#잘되나 확인
Attractive_Investment = Attractive_Investment.dropna()
#2. Quant portfolio #상위 40%이내로 짤라
rule = Attractive_Investment['Rank'].quantile(0.4)
Attractive_Investment = Attractive_Investment.iloc[:int(rule),:]

Attractive_Investment['V']=Rim
Attractive_Investment = Attractive_Investment.sort_values(by='V')
Attractive_Investment['Vrank'] = 1
Attractive_Investment['Vrank'] = Attractive_Investment['Vrank'].cumsum()#잘되나 확인

Attractive_Investment['Total'] = Attractive_Investment['Rank']*0.8+Attractive_Investment['Vrank']

Rank = Attractive_Investment['Total']

# Find the optimal portfolio
w = np.ones(len(Rank))
obj = lambda w: np.sum(w*Rank)
w0 = np.array([1]*len(Rank))/len(Rank)
cons = {'type': 'eq', 'fun': lambda w:  w.sum()-1}
y=100000
#저 함수가 1이 되게하는 것에서만 움직여라.
opt_w = optim.minimize(obj,w0, bounds=np.array([[0,min(0.05,0.1*y)]]*len(Rank)), constraints=cons).x
A = pd.DataFrame(opt_w)
A.index = Rank.index
A.columns = ['name']
A = A.sort_values(by='name')
Final = A.iloc[-25:]

###############################################################################
#1) Quant Portfolio의 투자성과
Kospi = file.parse('Kospi')
Stock = file.parse('Stock') #주식 시계열 데이터
Stock.index = Stock.pop('date')
Kospi.index = Kospi.pop('date')
Stock = np.exp(np.log(Stock).diff())-1
Kospi = np.exp(np.log(Kospi).diff())-1


Portfolio = pd.concat([Final,Stock.T],axis=1)
Portfolio.pop('name')
Portfolio = Portfolio.dropna(how='all')
Portfolio = Portfolio.mean()
Portfolio = Portfolio.cumsum()
Kospi = Kospi.cumsum()
Portfolio = pd.DataFrame(Portfolio,columns=['Portfolio'])
Data = pd.concat([Portfolio,Kospi],axis=1)
Data.plot()