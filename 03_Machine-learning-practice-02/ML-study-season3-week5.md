# 10/15 ML 스터디 시즌 3 5주 차

## 진행사항

* RNN 이론 공부
    * https://brunch.co.kr/@chris-song/9
    * https://www.slideshare.net/ByoungHeeKim1/recurrent-neural-networks-73629152
    * http://aikorea.org/blog/rnn-tutorial-3/
    * https://deeplearning4j.org/kr/lstm
    * http://solarisailab.com/archives/1451


## 차주 준비사항
* Back Propagation 에 대한 공부
* RNN 실습 - https://www.tensorflow.org/tutorials/recurrent 의 예제를 따라해보고 RNN 구성에 익숙해지기
* More -

### 각자 깨달은 점

#### Luca

* LSTM과 기본적인 RNN 모델의 차이를 이해함
  * Cell state의 동작 방식과 존재의의(?)를 조금 느껴봄
  * 정확히 장기 의존성 문제가 Cell state로 어떻게 해결되는지 학술적인 증명은 모르겠다. 깊이 갈수록 영향력이 적어짐에 따라 레이어별로 학습 여부(랄까 정도)를 결정할 수 있도록 하는 점이, 어렴풋이 Dropout 같다고 생각이 들었다.
* LSTM에서 GRU로 가는 과정이 Forget Gate와 Input Gate를 합친 식을 전체에 적용한 것과 비슷한 형태 같다고 생각함
  * Jay가 제시한 의견, "Cell state가 꼭 필요한가? 있어도 되지만 없어도 되지 않을까?"라는 의문에 GRU가 가능하다는 방향을 제시함
    * 그렇다면 Cell state 없이 장기 의존성 문제를 어떻게 해결한 것일까?
    * 없앤것이 아니라 Forget Gate와 Input Gate가 합쳐지고, Cell State와 Hidden State가 합쳐졌다고 [여기](https://brunch.co.kr/@chris-song/9)에서 말하고 있다. (잘 모르겠지만... 식을 풀어서 비교해보면 기본 LSTM의 Cell state가 H로 바뀌고 O(t)가 없어짐. 정확히 따지자면 아웃풋이 사라진 격?)
    * 그렇다면 기본 RNN 모델과는 어떤 차이에서 LSTM이라 부를 수 있으며, 장기 의존성 문제를 해결했을까?
* GRU에서 H het 계산 시, h(t-1)에 R(t)를 곱하고 있는데 이는 어떤 차이를 불러오는 걸까?
* 딥러닝을 공부할 수록 블랙박스는 많은데 하이퍼 파라미터도 많고, 각각의 영향성을 깨달을 만한 학문적 건덕지가 머리에 없으니 힘들다. 결국 하나하나 따져가다 보면 논문이 나오는데... 논문보다 레퍼런스 보다 시간 다갈거 같아서(볼거 같지도 않고...) CSE 딥러닝 강좌를 들어야하나 싶음. 결국 기본기가 탄탄해야할거 같다. 삽질은 필수

#### Gin

* Back Propagation 에 대한 이해가 선행되면 전반적인 이해도가 높아지겠다는 생각을 했음 - 공부합시다!
* 실습을 해보면 이걸로 뭘 할 수 있을지 느낌이 더 올듯 함

#### Brad

* 보다 수학적인 부분들과 증명에 대해 알아봐야 할 것 같다
* DP와의 유사성을 보면서 기존 알고리즘들이 ML에 영향을 줄 수 있는지 판단 필요

#### Jay
