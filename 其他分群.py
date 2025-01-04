from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation

df=pd.read_csv('Pokemon.csv', encoding='cp1252',header=0)
df_X = df[df.index<300]

Agglomerative = AgglomerativeClustering(n_clusters=5) #用 階層聚類 對 HP 到 Speed 的數據進行分群，分為 5 個群組。
Agglomerative.fit(df_X.loc[:,'HP':'Speed'])
print(Agglomerative.labels_)

X = df_X.loc[:,'HP':'Speed']
model = hierarchy.linkage(X,'ward') #計算連結矩陣 (linkage)，以 Ward 方法衡量群組間的距離。
#Ward 方法最小化群組內的平方誤差和。
#應用：幫助分析分群數的合理選擇（例如從樹狀圖中找出分割點）。
print(model)

hierarchy.dendrogram(model,orientation = 'top',labels = df_X.index)
ax = plt.gca()
bounds =ax.get_xbound()

ax.plot(bounds,[250,250],'--',c='k',lw=3)
ax.plot(bounds,[150,150],'--',c='k',lw=3)
plt.show()

lut = dict(zip(df_X['Type_1'].unique(),"rbg"))
row_colors = df_X['Type_1'].map(lut)
g = sns.clustermap(X,cmap = 'YlGnBu',row_colors = row_colors)
plt.setp(g.ax_heatmap.get_yticklabels(),rotation = 0)
plt.show()


X = df_X.loc[:,'HP':'Speed']
DBS = DBSCAN(eps = 35,min_samples = 2).fit(X) #使用 DBSCAN 進行密度聚類，設定 eps=35 為鄰域半徑，min_samples=2 為核心點的最小樣本數。
#DBSCAN 適合處理密度分布不均的數據，能識別噪聲（標籤為 -1）。
#應用：探測數據中的異常點（噪聲）或分散群組。
print(DBS.labels_)
print(pd.Series(DBS.labels_).value_counts())

X_std = StandardScaler().fit_transform(X)
AFF = AffinityPropagation() #使用 Affinity Propagation 進行親和傳播聚類，將數據標準化後進行聚類。
#Affinity Propagation 通過數據之間的相似性自動決定群組數量，不需要事先設定。
#應用：用於高維數據的快速分群。
labels = AFF.fit_predict(X_std)
print(labels)

'''
聚類分析：

聚類是無監督學習的一種，用於探索數據中的模式與結構。
應用於生物學（基因表達分析）、市場營銷（客戶分群）、圖像處理等領域。
各方法比較：
Agglomerative Clustering：
適合小型數據，結果受距離度量影響。

DBSCAN：
適合密度不均的數據，能識別異常點。
Affinity Propagation：
不需要預先設定群組數，適合自動分群。
'''