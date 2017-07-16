# 7/9 ML 스터디 시즌 2 7주 차

## 진행사항
- poker NN 
  - features one hot으로도 여전히 약 60%

## 차주 진행사항
- 정확도를 더 높여보기 
  - (예시) label의 0과 0이 아닌 것을 구분하여 모델 2개로 판단
 
## 후기

### Luca
* test set에 대해 정확도가 92%가 나와서 기뻐했다. 하지만 test set의 label을 잘못 잡았었다.
  * dropout이 적용되서 training set에 대한 정확도는 낮아도, test set에 대해서는 높을 수 있다고 생각했음
  * training set : 50%, test set : 92% 정도면 dropout을 input 0.8, hidden 0.7 에 비해 큰 차이라고 봐야하는 듯
  * 이런 경험들이 각 설정값과 결과에 대한 용납할 수 있는 결과인지를 체득하게 해준다고 느낌
* 스터디 시간 동안 학습 모델을 달리하여 테스트하는데, 노트북으로는 너무 느려서 500 steps 정도로 돌려서 5분에 겨우 하나 돌려보는 정도라 변화를 크게 못줌
* 결국 learning rate나 dropout, neuron 갯수를 달리하는 것에는 느린 학습 시간으로 허덕여서, ML 이외의 방법에 대해서 고민하였음
  * label을 0과 0이 아닌것으로 나누어 각각 학습하여, 2개의 모델로 먼저 0인지 아니라고 판단할 수 있는 유의미한 정확도를 구해 판단하고, 아니면 0이 아닌 모델로 판별해보자는 생각을 하였음
  * 하지만 0이 아닌가 맞는가의 모델의 신뢰도가 관건이라 생각함. 0인데 0이 아닌 모델에서 판별하면 오답이니
* jay로부터 learning cost는 label과 관계가 없다는 것에 대해 배움
* momentum에 대한 궁금증이 생김
  * https://tensorflow.blog/2017/03/22/momentum-nesterov-momentum/
* Adam Optimizer vs RMS optimizer : 차이가 궁금했다. 못찾았으나 일단 adam이 짱인걸로
  * https://stackoverflow.com/questions/36162180/gradient-descent-vs-adagrad-vs-momentum-in-tensorflow
  
### Gin
* data set의 특성을 보고 값 자체로 training 할지 class한 데이터로 봐서 one-hot encoding을 적용할지 잘 파악해야 겠다고 생각함
* neuron의 개수가 많아짐에 따라 점점 랩탑의 연산 속도가 한계가 오기 시작했음 (역시 장비인가!?)
* dropout을 적용해서 overfitting을 막아야 함을 배웠음


### Brad

### Jay
