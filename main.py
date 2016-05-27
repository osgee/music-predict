import pymysql
import numpy as np
import matplotlib.dates as mdates
import math
import datetime
import os
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

path = 'commits/'+datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

def show_user_actions():
    conn = pymysql.connect(host='localhost', port=3306, user='user', password='password', db='music')
    cur = conn.cursor()
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')
    sql1 = 'select ds,count(*) from user_action group by ds order by ds asc;'

    cur.execute(sql1)

    size = cur.rowcount

    data = [() for i in range(size)]

    i = 0
    for c in cur.fetchall():
        s = str(c[0])
        data[i] = (datetime.date(int(s[:4]), int(s[4:6]), int(s[6:])), c[1])
        i += 1

    r = np.rec.array(data, dtype=[('date', object), ('num','f')])

    fig, ax = plt.subplots()
    ax.plot(r.date, r.num)

    # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)

    datemin = datetime.date(r.date.min().year, 1, 1)
    datemax = datetime.date(r.date.max().year + 1, 1, 1)
    ax.set_xlim(datemin, datemax)

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()
    plt.show()


def show_user_long_tail():
    conn = pymysql.connect(host='localhost', port=3306, user='user', password='password', db='music')
    cur = conn.cursor()
    cur.execute('select user_id, count(*) num from user_action group by user_id order by count(*) desc;')
    size = cur.rowcount
    data = [() for i in range(size)]
    i = 0
    for c in cur.fetchall():
        data[i] = (i, c[1])
        i += 1

    fig, ax = plt.subplots()
    r = np.rec.array(data, dtype=[('id', 'f'), ('num', 'f')])
    ax.plot(r.id, r.num)

    ax.set_xlim(0, size)
    plt.show()


def show_per_user_actions():
    conn = pymysql.connect(host='localhost', port=3306, user='user', password='password', db='music')
    cur = conn.cursor()
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')

    sql1 = 'select ds, count(*) from ( select * from ( select * from user_action where user_id = "13fefedac242925d4cada328c66ce87b") as oneuser where song_id="4f59545a48dcf42baedbf2e21e174d0f") perusersong group by ds order by ds asc;'

    cur.execute(sql1)

    size = cur.rowcount

    data = [() for i in range(size)]

    i = 0
    for c in cur.fetchall():
        s = str(c[0])
        data[i] = (datetime.date(int(s[:4]), int(s[4:6]), int(s[6:])), c[1])
        i += 1

    r = np.rec.array(data, dtype=[('date', object), ('num','f')])

    fig, ax = plt.subplots()
    ax.plot(r.date, r.num)

    # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)

    datemin = datetime.date(r.date.min().year, 1, 1)
    datemax = datetime.date(r.date.max().year + 1, 1, 1)
    ax.set_xlim(datemin, datemax)

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()
    plt.show()


def test_stationarity(timeseries):

    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

def time_series_per_artist(i):
    # % matplotlib
    # inline

    # rcParams['figure.figsize'] = 15, 6
    # data = pd.read_csv('AirPassengers.csv')
    # print(data.head())
    # print('\n Data Types:')
    # print(data.dtypes)
    # dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
    # data = pd.read_csv('AirPassengers.csv', parse_dates='Month', index_col='Month', date_parser=dateparse)
    # ts = data['#Passengers']
    # ts.head(10)
    # plt.plot(ts)
    # plt.show()

    with open('artplay.csv','r') as artplays:
        start = '3/1/2015'
        end = '8/31/2015'
        rng = pd.date_range(start, end, freq='D')
        artplays = artplays.readlines()[1:]
        artplay = artplays[i][artplays[i].index(',')+1:-1].split(',')
        artid = artplay[0]
        for c in range(len(artplay)-1):
            artplay[c+1] = int(artplay[c+1])
        ap = np.array(artplay[2:])
        ts = pd.Series(ap, index=rng)
        # plt.plot(ts)
        # plt.show()
        ts_log = np.log(ts)
        # plt.plot(ts_log)
        # plt.plot(ts)
        # fd = 20
        # moving_avg = pd.rolling_mean(ts_log, fd)
        # plt.plot(ts_log)
        # plt.plot(moving_avg, color='red')
        # ts_log_moving_avg_diff = ts_log - moving_avg
        # ts_log_moving_avg_diff.head(fd)
        # ts_log_moving_avg_diff.dropna(inplace=True)
        # test_stationarity(ts_log_moving_avg_diff)

        # ts_log_diff = ts_log - ts_log.shift()
        # plt.plot(ts_log_diff)

        # ts_log_diff.dropna(inplace=True)
        # test_stationarity(ts_log_diff)

        # lag_acf = acf(ts_log_diff, nlags=20)
        # lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

        # # Plot ACF:
        # plt.subplot(121)
        # plt.plot(lag_acf)
        # plt.axhline(y=0, linestyle='--', color='gray')
        # plt.axhline(y=-1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
        # plt.axhline(y=1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
        # plt.title('Autocorrelation Function')

        # Plot PACF:
        # plt.subplot(122)
        # plt.plot(lag_pacf)
        # plt.axhline(y=0, linestyle='--', color='gray')
        # plt.axhline(y=-1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
        # plt.axhline(y=1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
        # plt.title('Partial Autocorrelation Function')
        # plt.tight_layout()

        # AR Model
        # model = ARIMA(ts_log, order=(2, 1, 0))
        # results_AR = model.fit(disp=-1)
        # plt.plot(ts_log_diff)
        # plt.plot(results_AR.fittedvalues, color='red')
        # plt.title('RSS: %.4f' % sum((results_AR.fittedvalues - ts_log_diff) ** 2))


        # MA Model
        # model = ARIMA(ts_log, order=(0, 1, 2))
        # results_MA = model.fit(disp=-1)
        # plt.plot(ts_log_diff)
        # plt.plot(results_MA.fittedvalues, color='red')
        # plt.title('RSS: %.4f' % sum((results_MA.fittedvalues - ts_log_diff) ** 2))

        # Combined Model
        # order=(2, 1, 2)
        # model = ARIMA(ts_log, order=(2, 1, 2))
        # results_ARIMA = model.fit(disp=-1)

        # plt.plot(ts_log_diff)
        # plt.plot(results_ARIMA.fittedvalues, color='red')
        # plt.title('RSS: %.4f' % sum((results_ARIMA.fittedvalues - ts_log_diff) ** 2))

        # Taking it back to original scale
        # predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
        # print(predictions_ARIMA_diff.head())
        #
        # predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
        # print(predictions_ARIMA_diff_cumsum.head())
        #
        # predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
        # predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
        # predictions_ARIMA_log.head()

        # predictions_ARIMA = np.exp(predictions_ARIMA_log)
        # plt.plot(ts_log)
        # plt.plot(predictions_ARIMA_log)
        # plt.title('RMSE: %.4f' % np.sqrt(sum((predictions_ARIMA - ts) ** 2) / len(ts)))

        # decomposition = seasonal_decompose(ts_log)

        # trend = decomposition.trend
        # seasonal = decomposition.seasonal
        # residual = decomposition.resid
        #
        # plt.subplot(411)
        # plt.plot(ts_log, label='Original')
        # plt.legend(loc='best')
        # plt.subplot(412)
        # plt.plot(trend, label='Trend')
        # plt.legend(loc='best')
        # plt.subplot(413)
        # plt.plot(seasonal, label='Seasonality')
        # plt.legend(loc='best')
        # plt.subplot(414)
        # plt.plot(residual, label='Residuals')
        # plt.legend(loc='best')
        # plt.tight_layout()


        # -*- coding: utf-8 -*-
        # arima时序模型

        # import pandas as pd

        # 参数初始化
        # discfile = 'arima_data.xls'
        forecastnum = 60

        # 读取数据，指定日期列为指标，Pandas自动将“日期”列识别为Datetime格式
        # data = pd.read_excel(discfile, index_col=u'日期')
        data = ts_log

        # 时序图

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        # ts.plot()
        # plt.show()

        # 自相关图
        from statsmodels.graphics.tsaplots import plot_acf
        # plot_acf(data).show()

        # 平稳性检测
        from statsmodels.tsa.stattools import adfuller as ADF
        # print(u'原始序列的ADF检验结果为：', ADF(data.ix[0]))
        # 返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore

        # 差分后的结果
        D_data = data.diff().dropna()
        # D_data.columns = [u'销量差分']
        # D_data.plot()  # 时序图
        # plt.show()
        # plot_acf(D_data).show()  # 自相关图
        # from statsmodels.graphics.tsaplots import plot_pacf
        # plot_pacf(D_data).show()  # 偏自相关图
        # print(u'差分序列的ADF检验结果为：', ADF(D_data[u'销量差分']))  # 平稳性检测

        # 白噪声检验

        print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(D_data, lags=1))  # 返回统计量和p值

        # 定阶
        # pmax = int(len(D_data) / 10)  # 一般阶数不超过length/10
        # qmax = int(len(D_data) / 10)  # 一般阶数不超过length/10
        pmax = 10
        qmax = 10
        bic_matrix = []  # bic矩阵
        for p in range(pmax + 1):
            tmp = []
            for q in range(qmax + 1):
                try:  # 存在部分报错，所以用try来跳过报错。
                    tmp.append(ARIMA(data, (p, 1, q)).fit().bic)
                except:
                    tmp.append(None)
            bic_matrix.append(tmp)

        bic_matrix = pd.DataFrame(bic_matrix)  # 从中可以找出最小值

        p, q = bic_matrix.stack().idxmin()  # 先用stack展平，然后用idxmin找出最小值位置。
        # p, q = 5, 5
        print(u'BIC最小的p值和q值为：%s、%s' % (p, q))
        model = ARIMA(data, (p, 1, q)).fit()  # 建立ARIMA(0, 1, 1)模型

        # predictions_ARIMA_diff = pd.Series(model.fittedvalues, copy=True)
        # predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
        # predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
        # predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
        # predictions_ARIMA = np.exp(predictions_ARIMA_log)

        # model.summary2()  # 给出一份模型报告
        res = model.forecast(forecastnum)  # 作为期5天的预测，返回预测结果、标准误差、置信区间。
        predictions = np.exp(res[0])

        rng_predict = pd.date_range('9/1/2015', '10/30/2015', freq='D')
        # pre_np = np.array(predictions)
        all_np = np.concatenate((artplay, np.asarray(predictions)))
        ts_pre = pd.Series(predictions, index=rng_predict)
        # plt.plot(ts_pre)
        # plt.plot(predictions_ARIMA)
        # plt.show()
        pres = np.asarray(predictions)
        write_to_file(artid, ts_pre)


def write_to_file(artid, ts):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+'/mars_tianchi_artist_plays_predict.csv', 'a+') as file:
        for i in range(ts.count()):
            file.write(artid + ',' + str(math.ceil(ts[i])) + ',' + ts.index[i].strftime('%Y%m%d')+'\n')


for i in range(50):
    time_series_per_artist(i)
# time_series_per_artist(1)
# show_per_user_actions()
# show_user_long_tail()