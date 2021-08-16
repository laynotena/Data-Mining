# 1.資料前處理
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def data_preprocess(data):
    # 1-c.NR表示無降雨，以0取代
    data = data.replace('NR',0) 
    # 1-b.缺失值以及無效值以前後一小時平均值取代
    #無法換做數值形式的空格以NaN填入
    data.iloc[:,0:] = data.iloc[:,0:].apply(pd.to_numeric,errors='coerce') 
    #把資料格式轉為1維陣列
    temp_data = pd.DataFrame(data.values.reshape(1,-1))
    #將空缺的數值以前一個小時的數值代替
    temp_f = temp_data.ffill(axis = 1)
    #將空缺的數值以後一個小時的數值代替
    temp_b = temp_data.bfill(axis = 1)
    #計算前後一小時的平均值
    data = (temp_f + temp_b) / 2 
    #回復資料格式
    data = pd.DataFrame(data.values.reshape(1656,24))
    return data

def train_test_split(data):
    train = data.iloc[:18*61,:] 
    test = data.iloc[18*61:,:]
    return train, test

def reshape_data(train, test):
    #1-e. 製作時序資料: 將資料形式轉換為行(row)代表18種屬性，欄(column)代表逐時數據資料
    #reshape的-1代表未知，reshape會自動調整成18列和相對應的欄位數量
    train = pd.DataFrame(train.values.reshape(18,-1)) 
    test = pd.DataFrame(test.values.reshape(18,-1))
    Index = ['AMB_TEMP','CH4','CO','NMHC','NO','NO2','NOx','O3','PM10','PM2.5','RAINFALL','RH','SO2','THC','WD_HR','WIND_DIREC','WIND_SPEED','WS_HR']
    train.index = Index 
    test.index = Index
    return train, test

def generate_data(data, label):
    # 2-a. 取6小時為一單位切割
    for i in range(data.shape[1]-6): 
        if label == 'pm':
            if i==0:
                X = data.iloc[:,0:6].values 
            else:
                temp = data.iloc[:,i:i+6].values
                X = np.concatenate((X,temp),axis=0)
        if label == 'all':
            if i==0:
                X = data.iloc[:,0:6].values
            else:
                temp = data.iloc[:,i:i+6].values
                X = np.concatenate((X,temp),axis=0)
    Y = data.iloc[:,6:].T.values
    return X, Y


def fit(X_train, X_test, label):
    X_train, Y_train = generate_data(X_train, label)
    X_test, Y_test = generate_data(X_test, label)
    #2-c.使用兩種模型 Linear Regression 和 Random Forest Regression 建模
    #LR
    lr = LinearRegression().fit(X_train, Y_train.ravel())
    lr_pred = lr.predict(X_test)
    # 2-d. 用測試集資料計算MAE
    lr_MAE = mean_absolute_error(Y_test.ravel(), lr_pred)
    print(label," LR: ",lr_MAE)
    #RF
    rf = RandomForestRegressor(n_estimators=30).fit(X_train, Y_train.ravel())
    rf_pred = rf.predict(X_test)
    # 2-d. 用測試集資料計算MAE
    rf_MAE = mean_absolute_error(Y_test.ravel(), rf_pred)
    print(label," RF: ",rf_MAE)
    return lr_MAE, rf_MAE

#讀取檔案
data = pd.read_excel("D://106年新竹站_20180309.xls")
data = pd.DataFrame(data)

# 1-a.取出10.11.12月資料
data = data.iloc[4914:6570,] 
data = data.drop(['日期','測站', '測項'],axis=1)
# 1-b, 1-c 資料前處理
data = data_preprocess(data)
# 1-d. 將資料切割成訓練集(10.11月)以及測試集(12月)
train, test = train_test_split(data)
# 1-e. 製作時序資料: 將資料形式轉換為行(row)代表18種屬性，欄(column)代表逐時數據資料
train, test = reshape_data(train, test)

# 2.時間序列
# 2. b-1 只有 PM2.5
Pm_X_train = train[train.index=='PM2.5'] 
Pm_X_test = test[test.index=='PM2.5'] 
# 2. a, c, d
fit(Pm_X_train, Pm_X_test, 'pm')
fit(train, test, 'all')

