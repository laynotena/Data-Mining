# RNN, LSTM [Python]

參考資料：
HUNG-YI LEE (李弘毅) Machine Learning (2017,Spring) [課程](https://www.youtube.com/watch?v=xCGidAeyS4M&t=467s&ab_channel=Hung-yiLee)、[投影片](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2017/Lecture/RNN.pdf)

## Recurrent Nerual Network
### Memory Cell
* RNN與一般神經網路相比多了 <font color = "6288BD">**Memory Cell**</font> ，可以記憶前面發生的事
(紀錄 hidden layer 值的稱為 Elman Network，紀錄 output 值的稱為 Jordan Network，後者被認為表現較好)

1. 在RNN建立初始就必須給予 <font color = "6288BD">**Memory Cell**</font> 初始值，假設初始值為 0，<font color = "F2BA48"> **input** </font> 就會多一組 <font color = "6288BD">**Memory Cell**</font> 的 <font color = "F2BA48"> **input** </font> ，從  ![](https://i.imgur.com/Z36dZgK.png =20x)  變成   ![](https://i.imgur.com/xg7E6MH.png =50x)

![](https://i.imgur.com/WbSZCqU.png =500x)

<br>

2. 接著，<font color = "6288BD">**Memory Cell**</font> 會隨著每一次不斷更新成前一組 <font color = "81AB70">**Neuron**</font> 的值，
下一組的 <font color = "F2BA48"> **input** </font> 就會是 ![](https://i.imgur.com/AwLZwJD.png =50x)

![](https://i.imgur.com/nLOZ4Y3.png =500x)

<br>

再下一組 <font color = "F2BA48"> **input** </font> 就會是 ![](https://i.imgur.com/tKdhwAg.png =50x)
   * Memory Cell 讓 input 每一組的順序 (sequence order) 都有相關聯，如果任意改動順序，結果也會跟著改變

![](https://i.imgur.com/OLnLpJO.png =500x)

<br>

3. 每一層 hidden layer都會有一個 memory cell 記錄前一次的結果，再進行下一次的計算

![](https://i.imgur.com/QXBRgbl.png)

<br>

4. 雙向RNN
* 把兩個 rnn 的 hidden layer 取出後，再放到一層 output layer 產生最後的結果

![](https://i.imgur.com/ToAnNvu.png =500x)

<br>

## Long Short-term Memory
### Input Gate, Forget Gate, Memory Cell, Output Gate
* LSTM 多了不同的架構
<font color = "F17A49"> **Input Gate** </font> 可以控制hidden layer 或是 output的值能不能被記錄到 <font color = "6288BD">**Memory Cell**</font>
<font color = "81AB70">**Forget Gate**</font> 可以控制 <font color = "6288BD">**Memory Cell**</font> 裡面的值是否要更新
<font color = "F2BA48"> **Output Gate** </font> 可以控制 <font color = "6288BD">**Memory Cell**</font> 裡的紀錄能不能在計算的時候被讀取

* LSTM 的 input 就會有 4 個 ( input, 控制 <font color = "F17A49"> **Input Gate** </font> 的訊號, 控制 <font color = "81AB70">**Forget Gate**</font> 的訊號, 控制 <font color = "F2BA48"> **Output Gate** </font> 的訊號 )，output 會有一個

![](https://i.imgur.com/gV5kovJ.png =500x)

<br>

### 計算流程

(sigmoid 會介於 0 ~ 1 之間，也代表 gate 開啟的訊號)
   
![](https://i.imgur.com/LpJBU1Q.png =500x)

1. input z 先通過一層 activation function 變成 ![](https://i.imgur.com/OCJhT3v.png =35x)
   input gate 的訊號 ![](https://i.imgur.com/PaSOkzE.png =20x)，先通過一層 sigmoid 變成 ![](https://i.imgur.com/VJ6leVs.png =40x) = 0 代表關閉，1 代表開啟
   我們會把 input 乘積在一起 ![](https://i.imgur.com/EprBO5r.png =90x)

2. forget gate 的訊號 ![](https://i.imgur.com/J46IVrJ.png =20x) 會先通過一層 sigmoid 變成 ![](https://i.imgur.com/hJjGbnw.png =40x) = 0 代表開啟， 1 代表關閉
我們會把 forget gate 和 memory cell 乘在一起 ![](https://i.imgur.com/FMlAInP.png =70x)

3. memory cell 的值會被更新 ![](https://i.imgur.com/KrA3GXK.png =180x)

4. memory cell 的值會再通過一層 activation function 變成 ![](https://i.imgur.com/720DZMp.png =40x)
output gate的訊號 ![](https://i.imgur.com/X9EBTCn.png =20x) 會先通過一層 sigmoid 變成 ![](https://i.imgur.com/FQjo4Ls.png =40x) = 0 代表關閉，1 代表開啟
我們會把 memory cell 和 output gate 乘在一起 ![](https://i.imgur.com/OMtUprw.png =100x)
當作 output a











