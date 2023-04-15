#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
import re
from collections import Counter


# In[ ]:




# In[7]:


# 创建一个字体对象
font = FontProperties(fname=r"D:\Chrome_downloads\字体合并补全工具-压缩字库-1.1.0-windows-x64\songTNR.ttf", size=20)
config = {   
    "mathtext.fontset":'stix',
    "font.family":'serif',
    "font.serif": ['SimSun'],
    "font.size": 20, #  字号，大家自行调节
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)
# 载入TimesSong（下载链接中），将'filepath/TimesSong.ttf'换成你自己的文件路径
SimSun = FontProperties(fname=r"D:\Chrome_downloads\字体合并补全工具-压缩字库-1.1.0-windows-x64\songTNR.ttf") 


# In[39]:


def preprocess(path):
    '''
    数据预处理函数
    args:
    path - QAR数据路径
    '''
    data = pd.read_excel(path)
    data_1 = data.iloc[2:].reset_index(drop=True)
    data_1['time'] = data_1.index.values #由于数据项是按时间排列的，可以构建时间特征项
    data_1[' GEAR SELECT DOWN'] = data_1[' GEAR SELECT DOWN'].map({'DOWN': 1}) #对起落架这一数据项进行编码
    data_1[" GEAR SELECT DOWN"].fillna(0, inplace = True) 
    data_1['A/T ENGAGED'] = data_1['A/T ENGAGED'].map({'DISENGD':0,'ENGAGED':1})
    data_1['ANY A/P ENGAGED'] = data_1['ANY A/P ENGAGED'].map({'OFF':0,'ON':1})
    #将数据中的True和False编码为0-1变量
    data_1 = data_1.replace({'False': 0, 'True': 1})
    bool_cols = data_1.select_dtypes(include='bool').columns  # 选取布尔型列
    data_1[bool_cols] = data_1[bool_cols].astype(int)  # 将布尔型列转换为整型

    #将起落机场编码为数字

    data_1['DEPARTURE AIRPORT'] = data_1['DEPARTURE AIRPORT'].apply(lambda x: int(re.findall('\d+', x)[0]))
    data_1['DESTINATION AIRPORT'] = data_1['DESTINATION AIRPORT'].apply(lambda x: int(re.findall('\d+', x)[0]))

    #data_1['DEPARTURE AIRPORT'] = data_1['DEPARTURE AIRPORT'].map({'机场68': 68}) 
    #data_1['DESTINATION AIRPORT'] = data_1['DESTINATION AIRPORT'].map({'机场117': 117}) 
    
    return data_1


# In[22]:


def Quality_analysis_1(data):
    '''
    数据质量可靠性分析：
    缺失值检测
    args:
    data - 预处理后的QAR数据
    '''
    null_percent = data.isna().mean()
    if sum(null_percent.values)!=0:
        print('数据缺失')
        return null_precent
    else:
        print('数据无缺失')
        return 0 


# In[64]:


def Quality_analysis_2(data,i):
    '''
    数据质量可靠性分析：
    离群值检测
    args:
    data - 预处理后的QAR数据
    '''
    pca = PCA(n_components=2) #对数据进行PCA降维
    pca.fit(data.iloc[:,3:64])
    X_pca = pca.transform(data.iloc[:,3:64])
    #利用DBSCAN算法进行聚类，也可以更换聚类方法
    dbscan = DBSCAN(eps=0.8, min_samples=2)
    dbscan.fit(X_pca)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan.labels_)
    plt.savefig(r"D:/2023年MathorCup高校数学建模挑战赛赛题/D题/fig/"+str(i)+".png",dpi=300)
    plt.show()
    # 计算异常值比例
    outlier_ratio = np.sum(dbscan.labels_ == -1) / len(dbscan.labels_)
    print('异常值比例：',i, outlier_ratio)


# In[66]:


def Importance_analysis(data_1):
    '''
    数据重要程度分析
    args:
    data_1 - 预处理后的QAR数据
    '''
    data_1['Max_G'] = data_1.iloc[:,14:24].max(axis=1) #构造每秒钟的最大着陆G值作为预测变量
    y = data_1['Max_G']
    X = data_1.iloc[:,3:65] #其余特征为自变量
    rf = RandomForestRegressor(n_estimators=100) #构建随机森林回归模型
    rf.fit(X, y)
    importances = rf.feature_importances_ #输出特征重要程度
    indices = np.argsort(importances)[::-1] #按照重要程度对特征变量进行排序并输出
    for f in range(X.shape[1]):
        print("%d. %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))


# In[67]:


path = r"D:/2023年MathorCup高校数学建模挑战赛赛题/D题/附件/附件1 关键参数的航段数据/1.xlsx"
data_1 = preprocess(path)


# In[5]:


data_1.head(5)


# In[10]:


Quality_analysis_1(data_1)


# In[23]:


for i in range(0,8):
    path = r"D:/2023年MathorCup高校数学建模挑战赛赛题/D题/附件/附件1 关键参数的航段数据/" + str(i+1)+ '.xlsx'
    data_1 = preprocess(path)
    a = Quality_analysis_1(data_1)


# In[65]:


for i in range(0,8):
    path = r"D:/2023年MathorCup高校数学建模挑战赛赛题/D题/附件/附件1 关键参数的航段数据/" + str(i+1)+ '.xlsx'
    data_1 = preprocess(path)
    a = Quality_analysis_2(data_1,i)


# In[68]:


sns.boxplot(x=data_1['COG NORM ACCEL.8'])


# In[71]:


sns.boxplot(x=data_1['COG NORM ACCEL.2'])


# In[72]:


sns.boxplot(x=data_1['COG NORM ACCEL.9'])


# In[ ]:




