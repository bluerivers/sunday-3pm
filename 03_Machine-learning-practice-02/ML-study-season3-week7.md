# 10/29 ML 스터디 시즌 3 7주 차

## 진행사항


## 차주 준비사항

* 이론 - RNN, LSTM 공부하고 이해하기
  * 동영상 시리즈 - [RNN Introduction](https://www.youtube.com/watch?v=yETQCIyggjY&index=11&list=PL1H8jIvbSo1q6PIzsWQeCLinUj_oPkLjc) 부터 RNN 학습 메커니즘까지 파악하기
* 실습
  * Back Propagation Numpy로 구현하기
  * RNN 관련 실습 찾아오기 (해오면 더욱 좋음)


### 각자 깨달은 점

#### Luca

* Back propagation의 증명을 억지로 이해한 느낌이 듬. N Neuron에서의 loss는 N-1, N-2, ... 등의 loss로 이어지는 점화식인 것을 이해함. 계속 역으로 영향력(loss)이 전파되어 매번 각 퍼셉트론(=Neuron)에서 매번 미분을 계산 안해도 되기에 매번 loss를 계산하면 몇만년 걸릴게 줄어든다로 말하는데 이게 크게 와닿지가 않음. 컴퓨터를 어릴 때부터 접한 사람으로써는 없던 시절을 체감하지 못하는건지. 그리고 역전파라고 했는데, 번역체라서 생긴 역이 아닌 명백한 Back이 있다. 아무튼 이 Back이 무엇을 기준으로 Back인지를 모르겠다.
* RNN과 LSTM의 크나큰 차이는 망각, Forget gate의 여부인듯함. 모든 뉴런에서 인풋에 대해 다 영향력을 가지면 vanishing gradient 문제가 발생하여, 뒤 뉴런으로 갈수록 영향력이 작아진다고 이해함

#### Gin

* 증명을 손으로 해봄으로써 좀더 이해하게 됨. 이제 코드로 작성해서 이헤도를 더 높여보자! 역시 눈으로 머리로만 하려고 하면 안 됨 ㅠㅠ
* RNN 도 강의로 들어서 이론적으로 이해하는 것을 해봐야겠음. 단순히 따라하는 것과 이해하고 하는 것은 다를 듯함
* Special Thanks to 차영록 느님


#### Brad

*

#### Jay

* back prop은 단순히 chain rule이 맞음, 수학 공부 좀 해야할 듯 (편미분 헷갈린다)
