# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 17:14:10 2022

@author: Jay9696
"""

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

lr = LinearRegression()
Standard = StandardScaler()
def adj_r2(r2, n, k):
    return 1 - (n-1)*(1-r2)/(n-k-1)
def figure(x,y): #繪圖
    x = Standard.fit_transform(x)
    y = Standard.fit_transform(y)
    lr.fit(x,y)
    def reg_r2_mse(x, y, deg):
        # 產生 deg 次多項式特徵
        pol_d = PolynomialFeatures(degree=deg)
        x = pol_d.fit_transform(x)
        lr = LinearRegression()
        lr = lr.fit(x, y)
        y_pred = lr.predict(x)
        r2_lst.append(adj_r2(r2_score(y, y_pred), x.shape[0], 1))
        y_plot_lst.append(lr.predict(pol_d.fit_transform(x_plot)))
        
    r2_lst, mse_lst, y_plot_lst = [None], [None], [None]
    # 產生繪圖的 x 座標
    x_plot = np.linspace(x.min(), x.max(), 10).reshape(-1,1)
    reg_r2_mse(x, y, 1)
    reg_r2_mse(x, y, 2)
    reg_r2_mse(x, y, 3)
  
    plt.scatter(x, y, label='Training points', alpha=.4)
    plt.plot(x_plot, y_plot_lst[1], 
         color='red', lw=3, linestyle=':', 
         label='Polynomial (d=1), $R^2=%.2f$' % r2_lst[1])

    plt.plot(x_plot, y_plot_lst[2],
         color='gold', lw=3, linestyle='-',
         label='Polynomial (d=2), $R^2=%.2f$' % r2_lst[2])

    plt.plot(x_plot, y_plot_lst[3],
         color='blue', lw=3, linestyle='--',
         label='Polynomial (d=3), $R^2=%.2f$' % r2_lst[3])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
def train_test(x,y):
    
    x = Standard.fit_transform(x)
    y = Standard.fit_transform(y)
    
    lr.fit(x,y)
    '''
    print('w_1 =', lr.coef_[0])
    print('w_0 =', lr.intercept_)
    '''
    #令前60%數據為訓練集,後面40%為測試集
    X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.4,random_state=0)
    lr.fit(X_train,y_train)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    lr.fit(X_train, y_train)
    y_train_pred = lr.predict(X_train)
    y_test_pred = lr.predict(X_test)
    #求出MSE
    print('MSE(training): %.5f, MSE(testing): %.5f' %( 
    mean_squared_error(y_train, y_train_pred), 
    mean_squared_error(y_test, y_test_pred)))
    #求出RMSE
    print('RMSE(training): %.5f, RMSE(testing): %.5f' %( 
    mean_squared_error(y_train, y_train_pred)**0.5, 
    mean_squared_error(y_test, y_test_pred)**0.5))
    #求出R平方
    print('R^2(training): %.5f, R^2(testing): %.5f' %( 
    r2_score(y_train, y_train_pred), 
    r2_score(y_test, y_test_pred)))
    
    
    
    
   
it=input() #輸入想要看的遊戲
df=pd.read_csv('Twitch_game_data.csv', encoding='cp1252')
Games = df['Game'] == it
#先繪圖,求出pearson相關係數
plt.figure()
sns.heatmap(df[Games].corr(),annot=True)
plt.show()

input_x = input() #輸入X軸數據
input_y = input() #輸入Y軸數據
train_x = df[Games].loc[:, [input_x]]
train_y = df[Games].loc[:, [input_y]]
figure(train_x,train_y)
train_test(train_x,train_y)

print(input_x,'最大值 = ',train_x.max().to_string(index=False),'  最小值 = ',train_x.min().to_string(index=False),
      '  平均值 = ',train_x.mean().to_string(index=False),'  標準差 = ',train_x.std().to_string(index=False),
      '  變異數 = ',train_x.var().to_string(index=False),'  中位數 = ',train_x.median().to_string(index=False))
print(input_y,'最大值 = ',train_y.max().to_string(index=False),'  最小值 = ',train_y.min().to_string(index=False),
      '  平均值 = ',train_y.mean().to_string(index=False),'  標準差 = ',train_y.std().to_string(index=False),
       '  變異數 = ',train_y.var().to_string(index=False),'  中位數 = ',train_y.median().to_string(index=False))



'''
train_test(train_x,train_y)
'''

# 求出皮爾森相關係數圖


'''figure(df[Games].loc[:, ['Hours_watched','Hours_Streamed']],Games_streamer)'''
'''train_test(df[Games].loc[:, ['Hours_watched','Hours_Streamed']],Games_streamer)'''
   
'''
Games_hours_watched=df[Games].loc[:, ['Hours_watched']]
Games_hours_streamed=df[Games].loc[:, ['Hours_Streamed']]
Games_streamer=df[Games].loc[:, ['Streamers']]
figure(Games_streamer,Games_hours_watched)
train_test(Games_streamer,Games_hours_watched)
'''
#regressor = LinearRegression()
#regressor.fit(X_train, y_train)
    ## 計算出截距值與係數值
'''
    w_0 = regressor.intercept_
    w_1 = regressor.coef_
    print('Interception : ', w_0)
    print('Coeficient : ', w_1)
''' 
