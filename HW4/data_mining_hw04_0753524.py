import pandas as pd
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense,Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt

def data_generate(data):
    label = []
    text = []
    for i in range(len(data)):
        label.append(data.emoticon[i][0])
        text.append(data.emoticon[i][9:])
    df = pd.DataFrame(data = label, columns=['label'])
    df['text'] = text
    return df

def plot(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
class RNN():
    def __init__(self, input_dim=3800, output_dim=32, input_length=380, Drop_out=0):
        # 設置 Drop_out 參數
        self.DropOut = Drop_out
        # 呼叫 keras 提供的 sequential model
        self.rnn = Sequential()
        # 加入 input layer，因為是文本數據，所以使用 Keras提供的嵌入層 Embedding
        self.rnn.add(Embedding(input_dim = input_dim, output_dim = output_dim, input_length = input_length ))
        # 加入 Dropout
        self.rnn.add(Dropout(self.DropOut))
    def build_in(self):
        # 加入 SimpleRNN
        self.rnn.add(SimpleRNN(units=16))
        # 再多加一層 全連接層 NN
        self.rnn.add(Dense(units=256,activation='relu'))
        # 加入 Dropout
        self.rnn.add(Dropout(self.DropOut))
        # Add output layer
        self.rnn.add(Dense(units=1,activation='sigmoid'))
        #顯示模型架構
        self.rnn.summary()
    def fit(self,X_train, Y_train):
        # 編譯: 選擇損失函數、優化方法及成效衡量方式
        self.rnn.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        # 進行訓練, 訓練過程會存在 train_history 變數中
        self.train_history = self.rnn.fit(X_train,Y_train,epochs=10,batch_size=100,verbose=2,validation_split=0.2)
        #畫圖 acc
        plot(self.train_history,'accuracy','val_accuracy')
        #畫圖 loss
        plot(self.train_history,'loss','val_loss')
    def evaluation(self, X_test, Y_test):
        # 顯示訓練成果(分數)
        self.scores = self.rnn.evaluate(X_test, Y_test, verbose=1)
        print('the accuracy for RNN is : ', self.scores[1])

class lstm():
    def __init__(self, input_dim = 3800, output_dim = 32, input_length = 380, Drop_out = 0):
        self.Drop_out = Drop_out
        self.lstm = Sequential()
        self.lstm.add(Embedding(input_dim = input_dim, output_dim = output_dim, input_length = input_length))
        self.lstm.add(Dropout(0.7))
    def build_in(self):
        self.lstm.add(LSTM(32))
        self.lstm.add(Dense(units=256,activation='relu'))
        self.lstm.add(Dropout(self.Drop_out))
        self.lstm.add(Dense(units=1,activation='sigmoid'))
        self.lstm.summary()
    def fit(self, X_train, Y_train):
        self.lstm.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        self.train_history = self.lstm.fit(X_train,Y_train,epochs=10,batch_size=100,verbose=2,validation_split=0.2)
        plot(self.train_history,'accuracy','val_accuracy')
        plot(self.train_history,'loss','val_loss')
    def evaluation(self, X_test, Y_test):
        self.score = self.lstm.evaluate(X_test, Y_test, verbose=1)
        print('the accuracy for LSTM is : ', self.score[1])


# 1.資料前處理
training = pd.read_csv('training_label.txt', delimiter = "\t",names = ['emoticon'])
testing = pd.read_csv('testing_label.txt',delimiter = "\t", names = ['emoticon'])
# 1-a.建立 train, test 之 DataFrame
train = data_generate(training)
test = data_generate(testing)
Y_train = train['label'].astype(int)
Y_test = test['label'].astype(int)
# 1-b.建立Token
# Tokenizer 可以建立字典，把文字轉換為序列或向量
# 先建立一個 Tokenizer，預設3800個字
token = Tokenizer(num_words = 3800)
# 根據 texts更新字典
token.fit_on_texts(train.text)
# token.word_index
# texts_to_sequences文本轉換
x_train_seq = token.texts_to_sequences(train.text)
x_test_seq = token.texts_to_sequences(test.text)
# 1-c.統一資料長度：Padding截長補短
X_train = sequence.pad_sequences(x_train_seq, maxlen=380)
X_test = sequence.pad_sequences(x_test_seq, maxlen=380)


# 2.RNN
# DropOut = 0
model_RNN = RNN()
# 2-a.
model_RNN.build_in()
# 2-c.
model_RNN.fit(X_train, Y_train)
# 3.模型評估
model_RNN.evaluation(X_test, Y_test)
# 2-b.DropOut = 0.7
model_RNN_Drop = RNN(Drop_out = 0.7)
model_RNN_Drop.build_in()
model_RNN_Drop.fit(X_train, Y_train)
model_RNN_Drop.evaluation(X_test, Y_test)


#2.LSTM
# DropOut = 0
model_LSTM = lstm()
model_LSTM.build_in()
model_LSTM.fit(X_train, Y_train)
model_LSTM.evaluation(X_test, Y_test)
# DropOut = 0.7
model_LSTM_Drop = lstm(Drop_out = 0.7)
model_LSTM_Drop.build_in()
model_LSTM_Drop.fit(X_train, Y_train)
model_LSTM_Drop.evaluation(X_test, Y_test)


