# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 21:58:36 2022

@author: Jay9696
"""

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

df=pd.read_csv('Pokemon_new.csv', encoding='cp1252',header=0)

sensitive = 10
epsilon = 1.0
print('最大數')
print(df['Attack'].max())
print('最小數')
print(df['Attack'].min())
print('平均數')
print(df['Attack'].mean())
#print(df.loc[:, 'Height_m':'Catch_Rate'].mean())
print('標準差')
print(df['Attack'].std())
print('變異數')
print(df['Attack'].var())
print('中位數')
print(df['Attack'].median())



x=df['Attack']+np.random.laplace(loc=0,scale=sensitive/epsilon)
df['Attack'] = DataFrame(x)


print()
print()
#print(df[df['Catch_Rate'] >=50].shape[0])
#print(df['Catch_Rate']+np.random.laplace(loc=0,scale=sensitive/epsilon))
#print(np.random.laplace(loc=0,scale=sensitive/epsilon))
df.to_csv('pokemon_test.csv',index=False, encoding='cp1252')

df_test=pd.read_csv('Pokemon_test.csv', encoding='cp1252',header=0)
print('最大數')
print(df_test['Attack'].max())

print('最小數')
print(df_test['Attack'].min())
print('平均數')
print(df_test['Attack'].mean())
#print(df.loc[:, 'Height_m':'Catch_Rate'].mean())
print('標準差')
print(df_test['Attack'].std())
print('變異數')
print(df_test['Attack'].var())
print('中位數')
print(df_test['Attack'].median())

