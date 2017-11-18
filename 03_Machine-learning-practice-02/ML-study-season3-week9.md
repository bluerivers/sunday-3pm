# 11/12 ML 스터디 시즌 3 9주 차

## 진행사항

* RNN Tensorflow 예제 이해
* https://github.com/golbin/TensorFlow-Tutorials/blob/master/README.md#10---rnn 예제 이해

## 차주 준비사항

* Auto Encoder & GAN 코드 실행/분석
    * https://github.com/golbin/TensorFlow-Tutorials/blob/master/08%20-%20Autoencoder/01%20-%20Autoencoder.py
    * https://github.com/golbin/TensorFlow-Tutorials/blob/master/09%20-%20GAN/01%20-%20GAN.py
    * https://github.com/golbin/TensorFlow-Tutorials/blob/master/09%20-%20GAN/02%20-%20GAN2.py
* 코드를 이해하기 위한 이론적 부분 공부하기

### 각자 깨달은 점

#### Luca

* RNN과 Back propagation을 그렇게 봤음에도 Seq2Seq 같은 실제 적용 모델이 나오니 연결이 안되었다. 역시 실전이 필요함을 깨달음
* Seq2Seq의 decoder에서 학습할때는 output를 통해 맞는지 확인해가는 과정이 있다. 하지만 테스트에서는 이 부분을 'PPPP'로 채우는데 이렇게 동작할때 output은 어떤 역할을 가지는지가 궁금함(P가 아닌 다른 문자를 넣어도 되는 건 알겠음)


#### Gin

* 역시 코드를 보고 동작을 이해하려고 하니 궁금한 점도 많이 생기고 이해도도 깊어지는 것 같음. 이론과 실습의 쌍끌이!


#### Brad

* 결석


#### Jay

* seq2seq 에서 decoder input에 왜 label 값을 넣어서 learning하는가? (test할 때는 decoder input이 있지도 않을텐데)
* 일단 decoder input에 <go><pad><pad>...<pad>를 넣고 학습해도 잘 되긴함 (우리 예제가 너무 작아서 성능에 어떤 영향을 주는지 알기는 어려움)
* decoder input 부분을 helper라고도 부르는거같고, label 대신 prediction값을 넣어주는 등 다른 방법도 사용되는거 같음
* https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/tutorials/seq2seq/
* https://github.com/tensorflow/nmt
