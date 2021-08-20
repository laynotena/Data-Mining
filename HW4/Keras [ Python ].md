# Keras [ Python ]
參考資料: 
1. [深度學習-Keras中的Embedding層的理解與使用](http://frankchen.xyz/2017/12/18/How-to-Use-Word-Embedding-Layers-for-Deep-Learning-with-Keras/)
2. [理解Keras参数 input_shape、input_dim和input_length](https://blog.csdn.net/pmj110119/article/details/94739765)
3. [使用Keras、Python、Theano和TensorFlow開發深度學習模型](https://cnbeining.github.io/deep-learning-with-python-cn/)
4. [神經網路中Epoch、Iteration、Batchsize相關理解和說明](https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/557816/)
5. [Keras 文檔](https://keras.io/zh/models/model/) (https://keras.io/api/)
6. [Keras 模型、函數及參數使用說明](https://ithelp.ithome.com.tw/articles/10191725)
7. [機器學習自學筆記09: Keras2.0](https://wenwu53.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E8%87%AA%E5%AD%B8%E7%AD%86%E8%A8%9809-keras2-0-3e5c9ac1658f)
8. [通過loss曲線診斷神經網路模型](https://www.twblogs.net/a/5e50ddf3bd9eee2117bfe333)

<br>

## Keras NN
### Sequential Model
* keras 提供了兩種 model
1. Functional API：支援多個輸入、多個輸出
2. Sequential Model (順序式模型)：就是一種簡單的模型，單一輸入、單一輸出，按順序一層(Dense)一層的由上往下執行。

<br>

### Dense
Dense 就是神經網路最常見的全連接層，output = activation( dot(input, kernel) + bias )
* input_shape 輸入維度
神經網路的第一層要給 input_shape，shape包含維度和長度或是用 input_dim 和 input_length 表示，之後的 Dense input_shape 都會接著前一層的 output_shape，不需要另外設置
```
model.add(Dense(units=32, input_shape(16,)))

model.add(Dense(units=32, input_dim = 3800, input_length =380 )
```
* units 輸出維度
* activation 激活函數

<br>

### Dropout
避免過度擬合，每次訓練都會按照比例拿走一部份神經元，只會在訓練的時候執行

### Compile
compile 編譯，選擇損失函數 loss、優化方法 optimizer 及成效衡量方式 metrics
* loss： 參考 [losses](https://keras.io/api/losses/)
* optimizer : 參考 [optimizers](https://keras.io/api/optimizers/)
* metrics: 參考[metrics](https://keras.io/api/metrics/)

### Summary 
* 顯示整個神經網路的架構

```python=
from keras.models import Sequential
from keras.layers.core import Dense,Dropout
# build
model = Sequential()
model.add(Dense(units = 32, input_dim = 3800, input_length =380, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu')
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()
# compile
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
```
train_history = model.fit(X_train, Y_train)
score = model.evaluate(X_test, Y_test, verbose=1)

<br>

### Fit
訓練模型
* epochs 總共要訓練幾輪
* batch_size 每次要訓練的量
* iteration : iteration = 樣本數量 / batch_size
* verbose 是顯示模式，0 = 安静模式, 1 = 進度條, 2 = 每輪一行
* validation_split 介於 0 ~ 1，要分多少比例到 validation
* fit 會回傳一個 train_history
train_history.history 可以看到每個 epoch 的 training accuracy, training loss, validation accuracy, validation loss

```python=
 train_history = fit(X_train,Y_train,epochs=10,batch_size=100,verbose=2,validation_split=0.2)
 ```

### Train History
* 把 training_history 的 training loss 和 validation loss畫出來，可以簡單判斷模型是否有 underfitting 或是 overfitting的問題

```python=
def plot(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
plot(train_history, 'accuracy', 'val_accuracy')
plot(train_history, 'loss', 'val_loss')
```

#### Underfit
* training loss下降的非常平緩以致於好像都沒有下降，這說明模型根本沒有從訓練集學到東西
![](https://i.imgur.com/yCmBxT3.png =400x)
* 訓練結束時候training loss還在繼續下降，這說明還有學習空間，模型還沒來得及學就結束了
![](https://i.imgur.com/JghNJCh.png =400x)


#### Overfit
* training loss一直在不斷地下降，而validation loss在某個點開始不再下降反而開始上升了，我們應該在這個拐點處停止訓練
![](https://i.imgur.com/wkdbjWP.png =400x)

#### Goodfit
* training loss和validation loss都已經收斂並且之間相差很小很小幾乎沒有肉眼的差距，而通常traing loss會更小，他們之間的gap叫做generalization gap。
![](https://i.imgur.com/B8kg8QZ.png =400x)

<br>

### Evaluate
* evaluate 會回傳一個 score ，紀錄誤差和 testing accuracy
```
score = evaluate(X_test, Y_test, verbose=1)
print('testing loss  : ', self.score[0])
print('testing accuracy : ', self.score[1])
```

<br>

## Keras Embedding
* Keras提供了一個嵌入層，適用於文本數據的神經網路，輸入的 input 為整數編碼，每個字都只能用一個唯一的整數表示，這個步驟可以使用 Keras 提供的 Tokenizer API 執行
* Keras Embedding 可以單獨使用來學習一個單詞嵌入，並保存在另一個模型中使用；或者作為深度學習模型的一部份，嵌入與模型本身一起學習；也可以把 pretrain 的字典嵌入模型


### Keras Tokenizer
* Tokenizer 可以建立字典，把文字轉換為序列或向量
```python=
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

# 先建立一個 Tokenizer，預設3800個字
token = Tokenizer(num_words = 3800)
# 根據 texts 更新字典
token.fit_on_texts(train.text)
# token.word_index
# texts_to_sequences文本轉換
x_train_seq = token.texts_to_sequences(train.text)
# 統一資料長度：Padding截長補短
x_train = sequence.pad_sequences(x_train_seq, maxlen=380)
```

<br>

* 嵌入層被定義為神經網路的第一層 hidden layer ，必须指定3個参數：input_dim (總詞彙數量)、output_dim (每個詞會被轉換成幾個維度的向量)、input_length (輸入詞彙的長度)
* 嵌入層會隨機初始化權重，嵌入訓練集中的所有詞彙並學習，如果將模型保存到文件中，嵌入層的權重也會一起被保存

* 如果要接 Dense 層在 Embedding 層後面，要注意如果 Embedding層不是一維向量，必須先使 Flatten 將 Embedding 層轉為一維向量
```python=
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()
```

###### tags: `Python`
