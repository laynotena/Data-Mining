import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import graphviz

#讀取檔案
data = pd.read_csv('D://character-deaths.csv')

#處理缺失值
data = data.fillna(0)


#將 Death Chapter 視為 Label
Y = data['Death Chapter']
Y = pd.DataFrame(np.where(Y>0,1,0))
Y.columns=['Actual Data']


#移除 label 資料，並把 Allegiances 欄位做 dummy
X = data.drop(['Name','Death Year','Book of Death','Death Chapter'],axis=1)
X = pd.get_dummies(X, columns=['Allegiances'])
#print(data['Allegiances'].value_counts())

#呼叫 scikit-learn 套件切分訓練集與測試集
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=345723)

#呼叫 scikit-learn 決策數分類器執行預測
clf = tree.DecisionTreeClassifier(max_depth=5)
clf = clf.fit(X_train,Y_train)
pred = clf.predict(X_test)
print('Accuracy of training set',clf.score(X_train,Y_train))

#呼叫 scikit-learn 混淆矩陣計算 accuracy, precision, recall
CF_M = pd.DataFrame(confusion_matrix(Y_test, pred), index=['Actual Alive','Actual Death'], columns=['Predicted Alive','Predicted Death'])
print(CF_M)
print('Testing Set')
print('Accuracy:',accuracy_score(Y_test, pred))
print('Precision:',precision_score(Y_test, pred))
print('Recall:',recall_score(Y_test, pred))

#呼叫 graphviz 套件畫圖
dot_data = tree.export_graphviz(clf,out_file=None, feature_names=X_test.columns, class_names=['Alive','Death'], max_depth = 5)
graph = graphviz.Source(dot_data)
graph









