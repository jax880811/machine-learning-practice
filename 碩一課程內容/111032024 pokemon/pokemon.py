# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 21:19:04 2022

@author: Jay9696
"""
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

#df=pd.read_csv('Pokemon.csv', encoding='cp1252',header=0)
#先把原本的資料讀入 直接做資料處理 做成新的csv檔案
#缺少的資料就用眾數

#imp = SimpleImputer(missing_values=np.nan, 
                    #strategy='most_frequent')
#data = imp.fit_transform(df)  # 填補後回傳陣列
#df_imp = pd.DataFrame(data, columns=df.columns)
#df = df_imp
#df.info()

#df.to_csv('Pokemon_new.csv',index=False, encoding='cp1252')
df=pd.read_csv('Pokemon_new.csv', encoding='cp1252',header=0)

regression = []
regression_pca = []
def Descriptive_Statistics():#敘述性統計
    print('最大數')
    print(df.loc[:, 'Type_1':'Catch_Rate'].max())
    print('最小數')
    print(df.loc[:, 'Type_1':'Catch_Rate'].min())
    print('平均數')
    print(df.loc[:, 'Type_1':'Catch_Rate'].mean())
    #print(df.loc[:, 'Height_m':'Catch_Rate'].mean())
    print('標準差')
    print(df.loc[:, 'Type_1':'Catch_Rate'].std())
    print('變異數')
    print(df.loc[:, 'Type_1':'Catch_Rate'].var())
    print('中位數')
    print(df.loc[:, 'Type_1':'Catch_Rate'].median())
    print('眾數')
    print(df.loc[:, 'Type_1':'Catch_Rate'].mode())
    print(df.describe())
def correlation():#相關性統計
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(12,12))
    sns.heatmap(df.corr(),annot=True)
    plt.show()
    corr = df.loc[:, 'Total':'Speed'].corr()
    plt.figure()
    ax = sns.heatmap(
            corr, vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True, annot = True
            )
    ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
            )
    plt.show()
def Linear_Regression():#線性迴歸模型: 目標項為Catch_Rate
    plt.style.use('fivethirtyeight')
    x, y = df.loc[:, ['Total','HP','Attack','Defense','Sp_Atk','Sp_Def','Speed']], df.loc[:, ['Catch_Rate']]
    lr = LinearRegression()
    lr.fit(x, y)
    print('w_1 =', lr.coef_[0])
    print('w_0 =', lr.intercept_)
    X = sm.add_constant(x.to_numpy())
    model = sm.OLS(y, X)
    result = model.fit()
    print('迴歸係數：', result.params)
    print(result.summary())
    Standard = StandardScaler()
    x = Standard.fit_transform(x)
    y = Standard.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=0)
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
    X = df.loc[:, 'Total':'Speed'].values
    y = df['Catch_Rate'].values

    pol_d = PolynomialFeatures(degree=2)
    X_poly = pol_d.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=0)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_train_pred = lr.predict(X_train)
    y_test_pred = lr.predict(X_test)
    

    y_train_resid = y_train_pred - y_train
    y_test_resid = y_test_pred - y_test 

    sns.residplot(y_train_pred, y_train_resid, lowess=True, 
              color="skyblue", label='Training data', 
              scatter_kws={'s': 25, 'alpha':0.7}, 
              line_kws={'color': 'b', 'lw':2})
    sns.residplot(y_test_pred, y_test_resid, lowess=True, 
              color="yellowgreen", label='Testing data', 
              scatter_kws={'s': 25, 'marker':'x'}, 
              line_kws={'color': 'g', 'lw':2})
    plt.xlabel('Predicted') #繪製殘差方圖
    plt.legend()
    plt.show()
def Logistic_Regression():
    print('邏輯迴歸模型')
    X, y = df.loc[:, 'Total':'Speed'], df['isLegendary']
    X=np.array(X)
    y=np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
    #d_class_weights = dict(enumerate(class_weights))
    logit = LogisticRegression()
    logit.fit(X_train, y_train)
    class_names = ['YES ', 'NO']
    disp = plot_confusion_matrix(logit, X_test, y_test, 
                             display_labels=class_names, 
                             cmap=plt.cm.Blues)
    plt.grid()
    plt.show()
    y_pred = logit.predict(X_test)
    print(classification_report(y_test, y_pred))
    test_score = logit.score(X_test, y_test) * 100
    print('邏輯迴歸 ACCURACY = ',test_score,'%')
    regression.append(test_score)
def SVM():
    print('SVM分類模型')
    X, y = df.loc[:, 'Total':'Speed'], df['isLegendary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
    scale = StandardScaler().fit(X_train)
    X_train_std = scale.transform(X_train)
    X_test_std = scale.transform(X_test)
    svm = SVC(kernel='rbf', class_weight='balanced')
    svm.fit(X_train_std, y_train)
    y_pred = svm.predict(X_test_std)
    print(classification_report(y_test, y_pred))
    test_score = svm.score(X_test, y_test) * 100
    print('svm ACCURACY = ',test_score,'%')
    regression.append(test_score)
def Random_Forest():
    print('Random_Forest分類模型')
    X, y = df.loc[:, 'Total':'Speed'], df['isLegendary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
    # 建立決策樹分類器
    clf = RandomForestClassifier(max_depth=3, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # 產生分類報告
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    test_score = clf.score(X_test, y_test) * 100
    importances = clf.feature_importances_
    std = np.std([t.feature_importances_ for t in clf.estimators_], axis=0)
    idx = np.argsort(importances)[::-1]

    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[idx], 
        yerr=std[idx], align="center")
    plt.xticks(range(X.shape[1]), labels=X.columns[idx])
    plt.xlim([-1, X.shape[1]])
    plt.ylim([0, 1])
    plt.show()
    print('RANDOM FOREST ACCURACY = ',test_score,'%')
    # 建立特徵選取器，門檻值預設為重要性的平均值
    selector = SelectFromModel(clf)
    selector.fit(X_train, y_train)
    print('門檻值 =', selector.threshold_)
    print('特徵遮罩：', selector.get_support())
    # 選出新特徵，重新訓練隨機森林
    X_train_new = selector.transform(X_train)
    clf.fit(X_train_new, y_train)
    X_test_new = selector.transform(X_test)
    y_pred = clf.predict(X_test_new)
    print(classification_report(y_test, y_pred))
    regression.append(test_score)
def KNN():
    print('KNN分類模型')
    X, y = df.loc[:, 'Total':'Speed'], df['isLegendary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
    # 特徵標準化
    scale = StandardScaler().fit(X_train)
    X_train_std = scale.transform(X_train)
    X_test_std = scale.transform(X_test)
    # 建立最近鄰模型
    neighbors = NearestNeighbors(n_neighbors=3).fit(X_train_std)
    new_poke = [[600,150, 50, 120, 80, 140, 60]]
    new_poke_std = scale.transform(new_poke)
    # 取出最近鄰的距離與索引值
    dist, idx = neighbors.kneighbors(new_poke_std)
    for d, i in enumerate(idx.ravel()):
        print(df.iloc[i, 1], np.array(X_train.iloc[i, :]), 
          '，標準化後的距離 = %.5f'% dist[0][d])
    knn = KNeighborsClassifier(n_jobs=-1)
    knn.fit(X_train_std, y_train)
    # 輸出預測結果
    print('預測結果:',knn.predict(new_poke_std))
    # 輸出預測結果的機率
    print('預測結果的機率:',knn.predict_proba(new_poke_std))
    y_pred = knn.predict(X_test_std)
    print(classification_report(y_test, y_pred))
    test_score = knn.score(X_test_std, y_test) * 100
    print('knn ACCURACY = ',test_score,'%')
    regression.append(test_score)
def Principal_Component_Analysis():
    print('PCA分群模型')
    cols = ['Total','HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed']
    scaler = StandardScaler().fit(df[cols])
    cols_std = scaler.transform(df[cols])
    df_std = pd.DataFrame(cols_std, columns=cols)
    print(df_std.describe())
    num_pc = 2
    pca = PCA(n_components=num_pc)
    pca.fit(df_std)
    
    loadings = pd.DataFrame(pca.components_, columns=cols)
    loadings.index = ['PC'+str(i+1) for i in range(num_pc)]
    print(loadings)
    print('pca.explained_variance:',pca.explained_variance_)
    print('pca.explained_variance_ratio:',pca.explained_variance_ratio_) # 解釋變異比例
    pc_scores = pd.DataFrame(pca.transform(df_std))
    pc_scores.columns = ['PC'+str(i+1) for i in range(num_pc)]
    pc_scores.plot(kind='scatter', x='PC1', y='PC2')
    plt.show()
    num_pc = 3
    pca = PCA(n_components=num_pc)
    pca.fit(df_std)

    loadings = pd.DataFrame(pca.components_, columns=cols)
    loadings.index = ['PC'+str(i+1) for i in range(num_pc)]
    print(loadings)
    num_pc = 4
    pca = PCA(n_components=num_pc)
    pca.fit(df_std)

    loadings = pd.DataFrame(pca.components_, columns=cols)
    loadings.index = ['PC'+str(i+1) for i in range(num_pc)]
    print(loadings)
    X, y = df.loc[:, 'Total':'Speed'], df['isLegendary']
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=0)
    pca = PCA(n_components=2) # 前兩個主成分
    x_train_std = StandardScaler().fit_transform(X_train)
    x_test_std = StandardScaler().fit_transform(X_test)
    x_train_pca = pca.fit_transform(x_train_std)
    x_test_pca = pca.fit_transform(x_test_std)
    print('邏輯迴歸PCA分類模型')
    lr = LogisticRegression() # 邏輯迴歸PCA
    lr.fit(x_train_pca, y_train)
    y_pred = lr.predict(x_test_pca)
    print(classification_report(y_test, y_pred))#svm pca
    test_score = lr.score(x_test_pca, y_test) * 100
    print('邏輯迴歸 PCA ACCURACY = ',test_score,'%')
    class_names = ['YES ', 'NO']
    disp = plot_confusion_matrix(lr, x_test_pca, y_test, 
                             display_labels=class_names, 
                             cmap=plt.cm.Blues)
    plt.grid()
    plt.show()
    regression_pca.append(test_score)
    print('SVM PCA分類模型')
    svm = SVC(kernel='rbf', class_weight='balanced')
    svm.fit(x_train_pca, y_train)
    y_pred = svm.predict(x_test_pca)
    print(classification_report(y_test, y_pred))
    test_score = svm.score(x_test_pca, y_test) * 100
    print('svm PCA ACCURACY = ',test_score,'%')
    regression_pca.append(test_score)
    print('RANDOM FOREST PCA分類模型')
    clf = RandomForestClassifier(max_depth=4, n_jobs=-1) #RANDOM FOREST PCA
    clf.fit(x_train_pca, y_train)
    selector = SelectFromModel(clf)
    selector.fit(x_train_pca, y_train)
    X_train_new = selector.transform(x_train_pca)
    X_test_new = selector.transform(x_test_pca)
    clf.fit(X_train_new, y_train)
    y_pred = clf.predict(X_test_new)
    print(classification_report(y_test, y_pred))
    test_score = clf.score(X_test_new, y_test) * 100
    print('RANDOM FOREST PCA ACCURACY = ',test_score,'%')
    regression_pca.append(test_score)
    print('KNN PCA分類模型')
    scale = StandardScaler().fit(x_train_pca)
    X_train_std = scale.transform(x_train_pca)
    X_test_std = scale.transform(x_test_pca)
    knn = KNeighborsClassifier(n_jobs=-1) #knn PCA
    knn.fit(X_train_std, y_train)
    y_pred = knn.predict(X_test_std)
    print(classification_report(y_test, y_pred))
    test_score = knn.score(X_test_std, y_test) * 100
    print('knn PCA ACCURACY = ',test_score,'%')
    regression_pca.append(test_score)
    pca_regression_type = ['Logistic_Regression_pca','SVM_pca','Random_Forest_pca','KNN_pca']
    x = np.arange(len(pca_regression_type))
    plt.figure(figsize=(10,6))
    plt.bar(x, regression_pca, color=['red', 'green', 'blue', 'yellow'])
    plt.xticks(x, pca_regression_type)
    plt.xlabel('pca_regression_type')
    plt.ylabel('accuracy')
    plt.title('pca')
    plt.show()
    
    
Descriptive_Statistics()
correlation()
Linear_Regression()
Logistic_Regression()
SVM()
Random_Forest()
KNN()
regression_type = ['Logistic_Regression','SVM','Random_Forest','KNN']
x = np.arange(len(regression_type))
plt.figure(figsize=(9,6))
plt.bar(x, regression, color=['red', 'green', 'blue', 'yellow'])
plt.xticks(x, regression_type)
plt.xlabel('regression_type')
plt.ylabel('accuracy')
plt.title('test score')
plt.show()
Principal_Component_Analysis()


#class_weights = compute_class_weight(class_weight = "balanced", classes= np.unique(X_train), y= X_train.reshape(-1))
#class_weights = compute_class_weight(class_weight = "balanced", classes= np.unique(X_test), y= X_test.reshape(-1))
#class_weights = compute_class_weight(class_weight = "balanced", classes= np.unique(y_train), y= y_train.reshape(-1))
#class_weights = compute_class_weight(class_weight = "balanced", classes= np.unique(y_test), y= y_test.reshape(-1))