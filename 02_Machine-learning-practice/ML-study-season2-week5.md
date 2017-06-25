# 6/25 ML 스터디 시즌 2 5주 차

## 진행사항
deep neural net을 해보았다

## 차주 진행사항
classification에 적합한 데이터로 진행
* 버섯 https://archive.ics.uci.edu/ml/datasets/Mushroom
* 포커 https://archive.ics.uci.edu/ml/datasets/Poker+Hand

## 후기

### Luca

### Gin
* softmax 보다는 안정적으로 결과가 나오는 편이었다. 이제 간단한 pre-processing과 regularization 등을 통해 80% 이상의 정확도를 보일 수 있게 됐다. 그 이상으로 정확도를 높일 수 있는 방법에 대해 연구해봐야겠다.
* 기본적인 데이터를 보는 방법에 대해 다른 practice를 보면서 어떻게 보는지 분석해봐야겠다.
    * pandas에서 제공하는 각 변수 별 상관관계 그래프 뽑기 -  https://pandas.pydata.org/pandas-docs/stable/visualization.html#scatter-matrix-plot
* 기본적인 ML에 대해 조금은 편해지게 됐다.
* batch normalization 에 대해 공부해봐야겠다.


### Brad

### Jay
* hidden layer에서는 activation function으로 neuron을 구성하고 마지막 layer에서
데이터를 어떻게 볼 지 결정하면 되는듯
(linear regression처럼 값 그대로 볼지 logistic regression처럼 one-hot으로 볼 지)
* 아마도 layer가 많을수록 복잡한 function이 되는거 같은데, 이로 인해 overfitting되기도 쉬운듯
(linear function은 아무리 learning해봐야 overfitting 되기 어려운데 얘는 non-linear한 정도가 크니까?)
* dropout이 overfitting을 막기위해 필요한 건 맞는거 같은데, classification의 경우
one-hot encoding하니까 그냥 쓰면 될 거 같은데, regression 할 때는 dropout rate에 따라
scale이 바뀔 거 같은데 (아마 데이터가 중간으로 모여서 표준편차가 바뀔듯? layer 개수에도 영향 받을듯)
보정이 필요하지 않을까 하는 의문이 있음
* 다음주 포커 데이터는 outlier같은게 없기 때문에 overfitting이 될 여지가 별로 없지 않을까?
