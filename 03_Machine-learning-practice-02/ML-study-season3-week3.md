# 09/10 ML 스터디 시즌 3 3주 차


## 진행사항

* mnist digit recognition 더 잘해보기
* https://www.kaggle.com/zalando-research/fashionmnist 해보기

## 차주 준비사항

* 벼락치기 방지를 위한 수요일을 중간 Checkpoint로 설정
* Dataset 검토 후 풀어오기
    * https://www.kaggle.com/neha1703/movie-genre-from-its-poster
    * https://www.kaggle.com/daavoo/3d-mnist
* 지금처럼 Classification 말고 실질적인 결과가 나오는 뭔가 해보는 것은 어떤가?

### 각자 깨달은 점

#### Luca


#### Gin

* Convolution Network 은 padding을 통해 이미지 사이즈를 줄이지 않는다. Max Pooling 은 Stride와 Kernel Size에 따라 사이즈를 줄인다. 이것을 하는 이유는 적당한 간격의 이미지를 줄여도 그 이미지에 대한 특징은 남기 때문에 계산량을 줄이기 위해 Max Pooling을 하는 것이다.
* Adam Optimizer 와 CNN 과 같은 각 개념에 대한 정확한 이해가 필요하다.


#### Brad


#### Jay
* weight과 bias의 initial value가 각 feature의 평균, 분산에 영향을 줄 것이다.
느낌적인 느낌으로는 각 layer에서의 평균, 분산도 learning의 결과로 얻어진다고 볼 수 있을 거 같다.
=> Batch normalization이 이것과 연관이 있다.
* batch normalization 보다 보니 back propagation에 대해서도 더 정확히 아는게 좋을거 같다.
(위키피디아 reference보면 관련된거 엄청 많음,
http://timvieira.github.io/blog/post/2017/08/18/backprop-is-not-just-the-chain-rule/
이것도 한번 보면 좋겠다)
* 통계학도 좀 알아야되는데 sample의 variance 구할때 m / (m-1)하는거 뭔지 까먹음
* 보니까 tf.layers.batch_normalization 이런게 그냥 있음
* regularization은 weight가 작으면서도 cost를 낮게하는 weight을 찾겠다는 것.
noise(outlier)에 의해 weight vector의 길이가 길어지면 learning에서
weight vector의 방향을 바꾸기가 어려워짐 (local optimum에 빠지는 거 같군).
어차피 learning이 계속되면 커질 weight은 커짐 (될놈될)
http://neuralnetworksanddeeplearning.com/chap3.html
* param tuning은 작은 training set 가지고 learning 해보면서 정함
