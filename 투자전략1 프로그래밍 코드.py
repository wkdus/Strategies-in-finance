
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def main():
    #####  file open  #####
    os.chdir(r'C:\Users\Hyunjeong Kang\OneDrive\과제\투자전략 연구')

    file = pd.ExcelFile(r'벨류 퀄리티 투자전략 데이터.xlsx')

    Stock = file.parse('주식가격(종가)')
    Shares = file.parse('상장주식수')
    Earnings = file.parse('당기순이익')
    Debt = file.parse('부채비율')
    Roa = file.parse('ROA')
    Pbr = file.parse('PBR')
    liquidityAsset = file.parse('유동자산')
    TotalDebt = file.parse('총부채')
    Per = file.parse('PER')
    Pcr = file.parse('PCR')
    Psr = file.parse('PSR')
    EbitCash = file.parse('영업현금흐름')
    SalesProfit = file.parse('매출액')
    SalesCost = file.parse('매출원가')
    Beta = file.parse('베타')
    
    Stock.index = Stock.pop("date")
    Shares.index = Shares.pop("date")
    Roa.index = Roa.pop("date")
    Pbr.index = Pbr.pop("date")
    Debt.index = Debt.pop("date")
    Earnings.index = Earnings.pop("date")
    liquidityAsset.index = liquidityAsset.pop("date")
    TotalDebt.index = TotalDebt.pop("date")
    Per.index = Per.pop("date")
    Pcr.index = Pcr.pop("date")
    Psr.index = Psr.pop("date")
    EbitCash.index = EbitCash.pop("date")
    SalesProfit.index = SalesProfit.pop("date")    
    SalesCost.index = SalesCost.pop("date")
    Beta.index = Beta.pop("date")

    Stock.index = Stock.index.map(lambda x: x.strftime('%Y-%m'))    
    Shares.index = Shares.index.map(lambda x: x.strftime('%Y-%m'))
    Earnings.index = Earnings.index.map(lambda x: x.strftime('%Y-%m'))
    Debt.index = Debt.index.map(lambda x: x.strftime('%Y-%m'))
    Roa.index = Roa.index.map(lambda x: x.strftime('%Y-%m'))
    Pbr.index = Pbr.index.map(lambda x: x.strftime('%Y-%m'))
    liquidityAsset.index = liquidityAsset.index.map(lambda x: x.strftime('%Y-%m'))
    TotalDebt.index = TotalDebt.index.map(lambda x: x.strftime('%Y-%m'))
    Per.index = Per.index.map(lambda x: x.strftime('%Y-%m'))
    Pcr.index = Pcr.index.map(lambda x: x.strftime('%Y-%m'))
    Psr.index = Psr.index.map(lambda x: x.strftime('%Y-%m'))
    EbitCash.index = EbitCash.index.map(lambda x: x.strftime('%Y-%m'))
    SalesProfit.index = SalesProfit.index.map(lambda x: x.strftime('%Y-%m'))
    SalesCost.index = SalesCost.index.map(lambda x: x.strftime('%Y-%m'))
    Beta.index = Beta.index.map(lambda x: x.strftime('%Y-%m'))
    
    ### 날짜 기준 ###
    date='2015-01'
    print('date: ',date)
    ### KOSPI
    Anotherfile = pd.ExcelFile(r'KOSPI.xlsx')
    Kospi = Anotherfile.parse('KOSPI')    
    Kospi.index = Kospi.pop("date")
    Kospi.index = Kospi.index.map(lambda x: x.strftime('%Y-%m'))
    Kospi = (np.exp(np.log(Kospi).diff())-1)
    Kospi = Kospi[date:]
    Kospi = pd.DataFrame(Kospi,columns=['KOSPI'])
    Kospi.plot()
    
    ### 함수 가동 ###
    VS19 = ValueStrategy19(date,Stock,Shares,Earnings)
    VS20 = ValueStrategy20(date,Stock,Roa,Pbr,Debt)
    VS21 = ValueStrategy21(date,Stock,Shares,liquidityAsset,TotalDebt,Earnings)
    VS22 = ValueStrategy22(date,Stock,Shares,Pbr)
    VS23 = ValueStrategy23(date,Stock,Shares,Per,Pbr,Psr,Pcr)
    QS25 = QualityStrategy25(date, SalesProfit, SalesCost,liquidityAsset,Stock,Shares)
    QS26 = QualityStrategy26(date,Pbr,SalesProfit,SalesCost,liquidityAsset,Stock,Shares)
    BS = BuffetStrategy(date,Pbr,SalesProfit,SalesCost,liquidityAsset,Stock,Beta)

    ### 그림 그리기
    All = pd.concat([Kospi*100,VS21,VS22,QS25,QS26,BS],axis=1)
    All.plot()
    AllCum = All.cumsum()
    AllCum.plot()
#    print('< 변동성과 수익률 계산 >')
#    ### 변동성과 수익률 계산
    print('Kospi')
    Calculate(Kospi)
    Calculate(VS19)
    print('VS20')
    Calculate(VS20)
    print('VS21')
    Calculate(VS21)
    print('VS22')
    Calculate(VS22)
    print('VS23')
    Calculate(VS23)
    print('QS25')
    Calculate(QS25)
    print('QS26')
    Calculate(QS26)
    print('BS')
    Calculate(BS)
#    
def Calculate(Kospi):
     print(Kospi.std(),'%')

    ### 만약 연 1회 리벨런싱(동일 가중 수익률)한다면...
#    # 주식만 연으로 바꿔도 될 것 같다. 왜냐하면 최종적으로 계산하는 것은 주식이므로.
#    Stock.index = pd.DatetimeIndex(Stock.date)
#    Stock.pop('date')
#    Stock.index = Stock.index.to_period(freq='M')
#    Stock = Stock.groupby([Stock.index]).last()



    
def ValueStrategy19(date,Stock,Shares,Earnings):
    """
    [19 그레이엄의 마지막 선물]
    매수 전략: PER 5 이하 + 부채비율 50% 이하 상위 30개 종목
    매도 전략: 연1회 리밸런싱
    """
    PerRatio = Stock*Shares/(Earnings*100000000)
    Ranking = pd.DataFrame(PerRatio.loc[date])
    Ranking = Ranking.dropna()
    Ranking = Ranking.sort_values(date)

    Per = dict()
    for i in range(len(Ranking)):
        if (float(Ranking.iloc[i])>0)and(float(Ranking.iloc[i]<=5)):
            Per[Ranking.index[i]] = float(Ranking.iloc[i])

    Per = pd.DataFrame(pd.Series(Per))
    Per.columns = ['per']
    Per = Per.sort_values('per')

    #매수: 상위 30개 종목만 투자  --- 수정필요
    Per = Per.iloc[0:29,:]

    BuyStocks = Stock[Per.index]
    BuyStocks = (np.exp(np.log(BuyStocks).diff())-1) #보유수익률(R)
    Allprofit = BuyStocks.sum(axis=1)
    Allprofit = Allprofit[date:]
    Allprofit = pd.DataFrame(Allprofit,columns=['VS19'])
    
    print('Value Strategy 19')

    return Allprofit

def ValueStrategy20(date,Stock,Roa,Pbr,Debt):
    '''
    [20 그레이엄의 마지막 선물 업그레이드]
    매수 전략:ROA 5% 이상, 부채비율 50% 이하인 기업
                PBR<0.2인 기업을 제외하고 낮은 기업부터 30개 매수
    매도 전략:연 1회 동일비중 리밸런싱
    '''
    # 정렬할 날짜 기준: 2001-01
    Stockone = pd.DataFrame(Stock.loc[date])
    Roa = pd.DataFrame(Roa.loc[date])
    Pbr = pd.DataFrame(Pbr.loc[date])
    Debt = pd.DataFrame(Debt.loc[date])

    Stockone.columns = ['Stock']
    Roa.columns = ['ROA']
    Pbr.columns = ['PBR']
    Debt.columns = ['Debt']

    for i in range(len(Debt)):
        if Debt.iloc[i][0]=='완전잠식':
            Debt.iloc[i]=np.NAN
        else:
            Debt.iloc[i] = float(Debt.iloc[i])


    Sum = pd.concat([Stockone,Roa,Pbr,Debt],axis=1)
    Sum = Sum.dropna()
    for i in range(len(Sum)):
        if Sum['ROA'][i]<5:
            Sum['ROA'][i]=np.NaN
        elif Sum['Debt'][i]>50:
            Sum['ROA'][i]=np.NAN
        elif Sum['PBR'][i]<0.2:
            Sum['PBR'][i]=np.NAN

    Sum = Sum.dropna(how='any')
    Sum2 = Sum.sort_values('PBR')
    Sum2 = Sum2.iloc[0:29,:]

    Investment = Stock[Sum2.index]
    Investment = (np.exp(np.log(Investment).diff())-1)
    Investment = Investment.sum(axis=1)
    Investment = Investment[date:]
    Investment = pd.DataFrame(Investment,columns=['VS20'])
    print('Value Strategy 20')

    return Investment

def ValueStrategy21(date,Stock,Shares,liquidityAsset,TotalDebt,Earnings):
    '''
    **************************************************************************
    [21 NCAV 전략]
    "기업의 청산가치가 시가총액보다 50%나 높은 주식을 매수하자"

    매수 전략: 순유동자산 = 유동자산-총부채 > 시가총액
                세후이익 > 0
                (20~30개, 기업 수가 부족할 경우 1개 기업에 자신의 최대 5%만 투자)
    매도 전략: 연 1회 리벨런싱
    '''
    NetLA = (liquidityAsset - TotalDebt)*1000 #단위맞추기
    MarCap = Shares*Stock

    NetLA_t = pd.DataFrame(NetLA.loc[date])
    MarCap_t = pd.DataFrame(MarCap.loc[date])
    Stock_t = pd.DataFrame(Stock.loc[date])
    NP_t = pd.DataFrame(Earnings.loc[date])

    NetLA_t.columns = ['LA']
    MarCap_t.columns = ['MC']
    Stock_t.columns = ['Stock']
    NP_t.columns = ['NP']

    All = pd.concat([NetLA_t,MarCap_t,Stock_t,NP_t],axis=1)

    for i in range(len(All)):
        if All['LA'][i]<=All['MC'][i]:
            All['LA'][i]=np.NAN
        elif All['NP'][i]<=0:
            All['NP'][i]=np.NAN

    All = All.dropna(how='any')
    All = All.sort_values('NP')
    # 그냥 모두 매수#    Sum2 = Sum2.iloc[0:29,:]

    Investment = Stock[All.index]
    Investment = (np.exp(np.log(Investment).diff())-1)
    Investment = Investment.sum(axis=1)
    Investment = Investment[date:]
    Investment = pd.DataFrame(Investment,columns=['VS21'])
    print('Value Strategy 21')
    return Investment
    
def ValueStrategy22(date,Stock,Shares,Pbr):
    '''
    [벨류 투자전략 22 소형주+저PBR전략]
    매수 전략: 소형주(시가총액 하위 20%주식)만 매수
                PBR이 가장 낮은 주식30개 매수
                단 PBR<0.2 주식은 제외
    매도 전략: 월1회 리밸런싱
    '''
    MarCap = Shares*Stock

    MarCap_t = pd.DataFrame(MarCap.loc[date])
    Stock_t = pd.DataFrame(Stock.loc[date])
    Pbr_t = pd.DataFrame(Pbr.loc[date])

    MarCap_t.columns = ['MC']
    Stock_t.columns = ['Stock']
    Pbr_t.columns = ['PBR']

    All = pd.concat([MarCap_t,Stock_t,Pbr_t],axis=1)

    for i in range(len(All)):
        if All['MC'][i]>All['MC'].quantile(0.2):
            All['MC'][i]=np.NAN
        elif All['PBR'][i]<0.2:
            All['PBR'][i]=np.NAN

    All = All.dropna(how='any')
    All = All.sort_values('PBR')

    # PBR이 낮은 30개만 매수
    All = All.iloc[0:29,:]

    Investment = Stock[All.index]
    Investment = (np.exp(np.log(Investment).diff())-1)
    Investment = Investment.sum(axis=1)
    Investment = Investment[date:]
    Investment = pd.DataFrame(Investment,columns=['VS22'])
    print('Value Strategy 22')

    return Investment

    
def ValueStrategy23(date,Stock,Shares,Per,Pbr,Psr,Pcr):
    '''
    " 조작하기 힘든 매출액을 사용한 PSR을 사용하자."
    PSR이 가장 낮은 1%는 빼고 분석을 진행했다
    PSR 3.2 이상 - Bad
    PSR 0.35 이하 - Good
    PSR 0.2 이하 - Very Good

    [22 강환국 슈퍼 가치 전략]
    PER, PBR, PCR, PSR 별로 순위를 부여하고 4개 순위를 더해,
    소형주(시가총액 하위 20%)에만 투자한다.
    50개에 분산투자
    연 1회 리벨런싱
    '''
    date = '2001-01'
    PER_t = pd.DataFrame(Per.loc[date])
    PBR_t = pd.DataFrame(Pbr.loc[date])
    PCR_t = pd.DataFrame(Pcr.loc[date])
    PSR_t = pd.DataFrame(Psr.loc[date])
    MarCap = Shares*Stock
    MarCap_t = pd.DataFrame(MarCap.loc[date])
    
    All = pd.concat([MarCap_t,PER_t,PBR_t,PCR_t,PSR_t],axis=1)
    All = All.dropna(how='any')
    All.columns = ['MC','PER','PBR','PCR','PSR']
    All = All.sort_values('PER')
    All['1']=1
    All['1'] = All['1'].cumsum()
    All['PER'] = All['1']

    All = All.sort_values('PBR')
    All['2']=1
    All['2'] = All['2'].cumsum()
    All['PBR'] = All['2']

    All = All.sort_values('PCR')
    All['3']=1
    All['3'] = All['3'].cumsum()
    All['PCR'] = All['3']

    All = All.sort_values('PSR')
    All['4']=1
    All['4'] = All['4'].cumsum()
    All['PSR'] = All['4']

    All.pop('1')
    All.pop('2')
    All.pop('3')
    All.pop('4')
    
    #소형주만 골라내기
    for i in range(len(All)):
        if All['MC'][i]>All['MC'].quantile(0.2):
            All['MC'][i]=np.nan
    All = All.dropna(how='any')

    All['Rank'] = All.sum(axis=1)
    All = All.sort_values('Rank')
    
    # PBR이 낮은 30개만 매수
    All = All.iloc[0:29,:]

    Investment = Stock[All.index]
    Investment = (np.exp(np.log(Investment).diff())-1)
    Investment = Investment.sum(axis=1)
    Investment = Investment[date:]
    Investment = pd.DataFrame(Investment,columns=['VS23'])
    print('Value Strategy 23')

    return Investment
    
def QualityStrategy25(date, SalesProfit, SalesCost,liquidityAsset,Stock,Shares):
    '''
    [26]     GP/A + 소형주
    '''    
    GP_A = (SalesProfit-SalesCost)/liquidityAsset
    GP_A_t = pd.DataFrame(GP_A.loc[date])
    MarCap = Shares*Stock
    MarCap_t = pd.DataFrame(MarCap.loc[date])

    All = pd.concat([MarCap_t,GP_A_t],axis=1)
    All.columns = ['MC','GP/A']

    for i in range(len(All)):
        if All['MC'][i]>All['MC'].quantile(0.2):
            All['MC'][i]=np.NAN

    All = All.dropna(how='any')
    All = All.sort_values('GP/A')
    All = All.iloc[0:29,:]

    Investment = Stock[All.index]
    Investment = (np.exp(np.log(Investment).diff())-1)
    Investment = Investment.sum(axis=1)
    Investment = Investment[date:]
    Investment = pd.DataFrame(Investment,columns=['QS25'])
    print('QualityStrategy25')

    return Investment    
    
def QualityStrategy26(date,Pbr,SalesProfit,SalesCost,liquidityAsset,Stock,Shares):
    '''
    [26]     PBR  GP/A  소형주(시가총액 하위 20%)
    '''
    GP_A = (SalesProfit-SalesCost)/liquidityAsset
    GP_A_t = pd.DataFrame(GP_A.loc[date])
    MarCap = Shares*Stock
    MarCap_t = pd.DataFrame(MarCap.loc[date])
    Pbr_t = pd.DataFrame(Pbr.loc[date])
    
    All = pd.concat([MarCap_t,GP_A_t,Pbr_t],axis=1)
    
    All.columns = ['MC','GP/A','PBR']

    for i in range(len(All)):
        if All['MC'][i]>All['MC'].quantile(0.2):
            All['MC'][i]=np.nan
        elif All['PBR'][i]<0.2:
            All['PBR'][i]=np.nan

    All = All.dropna(how='any')
    All = All.sort_values('GP/A')
    All = All.iloc[0:29,:]

    Investment = Stock[All.index]
    Investment = (np.exp(np.log(Investment).diff())-1)
    Investment = Investment.sum(axis=1)
    Investment = Investment[date:]
    Investment = pd.DataFrame(Investment,columns=['QS26'])    
    print('QualityStrategy26')


    return Investment
    
def BuffetStrategy(date,Pbr,SalesProfit,SalesCost,liquidityAsset,Stock,Beta):
    '''
    저PBR, 저beta, high-quality:GP/A
    '''
    GP_A = (SalesProfit-SalesCost)/liquidityAsset
    GP_A_t = pd.DataFrame(GP_A.loc[date])
    Pbr_t = pd.DataFrame(Pbr.loc[date])
    Beta_t = pd.DataFrame(Beta.loc[date])
    
    All = pd.concat([GP_A_t,Pbr_t,Beta_t],axis=1)
    All.columns = ['GP/A','PBR','Beta']

    for i in range(len(All)):
        if All['PBR'][i]<0.2:
            All['PBR'][i]=np.nan
    All = All.dropna(how='any')
    
    All = All.sort_values('GP/A')
    All['temp']= 1
    All['temp']=All['temp'].cumsum()
    All['GP/A'] = All['temp']

    All = All.sort_values('Beta')
    All['temp']= 1
    All['temp']=All['temp'].cumsum()
    All['Beta']=All['temp']

    All = All.sort_values('PBR')
    All['temp']= 1
    All['temp']=All['temp'].cumsum()
    All['PBR']=All['temp']
    
    All.pop('temp')
    All = All.sum(axis=1)
    All = All.sort_values()
    All = All.iloc[0:29]

    Investment = Stock[All.index]
    Investment = (np.exp(np.log(Investment).diff())-1)
    Investment = Investment.sum(axis=1)
    Investment = Investment[date:]
    Investment = pd.DataFrame(Investment,columns=['BS'])
    
    print('Buffet Strategy')
    return Investment
    
main()
