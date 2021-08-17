import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier

def data_preprocessing(data):
    #類別資料轉換
    data = pd.get_dummies(data, columns=['workclass','education','marital_status','occupation','relationship','race','sex','native_country'])
    #數值資料標準化
    for i in ['age',  'fnlwgt',  'education_num','capital_gain', 'capital_loss', 'hours_per_week']:
        temp = [data[i].values]
        data[i] = preprocessing.normalize(temp)[0]
    # Label轉換
    data['income'] = data['income'].replace({' <=50K':0," >50K":1})
    return data

#c.請寫自行撰寫function進行k-fold cross-validation(不可使用套件)並計算Accuracy
def K_fold_CV(k, data):
    size = data.shape[0]//k
    acc=[]
    for i in range(k):
        # b-1.切割測試集
        test_set = data[i*size:(i+1)*size]
        #其餘為訓練集
        train_set = pd.concat([data[0:i*size],data[(i+1)*size:]],ignore_index=True)
        X_train = train_set.drop(['income'],axis=1)
        Y_train = train_set['income']
        X_test = test_set.drop(['income'],axis=1)
        Y_test = test_set['income']
        # b.請使用Gradient Boosting進行分類
        GDBT = GradientBoostingClassifier()
        GDBT.fit(X_train, Y_train)
        # b-2. 輪流計算accuracy 
        acc.append(GDBT.score(X_test,Y_test))        
    print(acc)
    # b-3. 將k次平均作為 output
    return np.mean(acc)

data = pd.read_csv('D://data.csv')

# a.資料前處理
data = data_preprocessing(data)

#3. 請計算k=10的Accuracy
print ('Mean accuracy of 10-fold CrossValidation：',K_fold_CV(10, data))
