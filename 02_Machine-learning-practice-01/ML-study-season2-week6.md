# 7/2 ML 스터디 시즌 2 6주 차

## 진행사항
deep neural net을 또 해보았다
- poker 60%
- mushroom 99~100%

## 차주 진행사항
- poker 정확도 올리기 (현재 60% 정도)
  - 참고 http://neuroph.sourceforge.net/tutorials/PredictingPokerhands/Predicting%20poker%20hands%20with%20neural%20networks.htm 

## 후기

### Luca
* label 데이터가 내가 선택한 index에 있는지 3번은 살펴보아야 한다. 맨날 끝에 들어있진 않고, 알았다한들 내가 그렇게 데이터를 뽑아 내고 있는지 확신하되 확인하자. 머쉬룸 ㅂㄷㅂㄷ
* layer는 2~3개 이상이면 오히려 좋지 않은 결과가 나옴. 하나씩 천천히 늘려가보는게 좋을듯 함
* hidden layer의 maxtrix size = neural size 는 feature의 두배보다는 적고 feature보다 더 작을 수도 있음. 갓제이님의 메소드 참고
* 연속적인 숫자라고 해서 연속적인 숫자가 NN에 의미있는 영향(?)을 가질거라고 생각하지 않는게 좋다. 포커에서 배움
* tensorflow를 CPU로 쓰는 것은 손코딩과 유사한 효율성인듯 하다. GTX 1080에서는 체감상 20배 이상 좋은 듯하다.
* nvidia-docker는 linux 계열만 지원한다. 윈도우에서는 쌩으로 tensorflow를 설치해야 GPU를 사용할 수 있다.
  * http://goodtogreate.tistory.com/entry/GPU-TensorFlow-on-Window-10-TensorFlow-GPU%EB%B2%84%EC%A0%84-%EC%9C%88%EB%8F%84%EC%9A%B010-%EC%84%A4%EC%B9%98
  * http://jaejunyoo.blogspot.com/2017/02/start-tensorflow-gpu-window-10.html
* 코드를 재활용하면 큰 낭패를 본다. 제대로 "코딩"을 하고 있는지 생각해야함
* categorical value는 로지스틱 선형 회귀로 wide하게 처리하는게 보통인가?
  * Tensorflow에서 categorical value를 위한 SparseTensors object를 지원한다. 이를 NN에서 사용할 수 있는지 궁금함
    * https://www.tensorflow.org/api_guides/python/sparse_ops#SparseTensor
  * https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/tutorials/wide/
  * https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/tutorials/linear/overview.html
* linear와 DNN을 결합하여 wide & deep도 시도해보면 재밌을 거 같다.
  * https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/tutorials/wide_and_deep/

### Gin
* 생각 없는 복붙은 화를 초래한다. 역시나 중요한 것은 domain에 대해 이해하고 그에 맞게 데이터를 잘라서 분석하는 것이다.
* 텐서플로우에 익숙해졌다고 생각하지만 역시나 아직은 자유자재로 쓰기엔 무리가 있다.
* 데이터에 따라 one-hot encoding을 적용하는 것도 중요하다,
* NN을 돌리니 슬슬 데이터를 뽑는 것이 느려지기 시작했다.

### Brad

### Jay
* initial 값에 따라 결과가 꽤 달라지므로 initialization을 잘 해야 할 거 같다.
* feature를 one-hot encoding 해서 해보기 (one-hot 안하고 하면 straight에서 A,1,2,3,4 같은 거에는 좋은 영향을 줄 수도 있지만 J,Q,K,A,2 같은 경우엔 안좋을 수도 있을거같음)
* regularization 같은 걸 적용해봐야겠다.
* optimal (sub-optimal?)은 실험해보면서 찾아야하지만 보통 layer 3개면 일반적은 문제를 푸는데 문제가 없다고 한다.
neuron의 개수도 input neuron의 개수에 비해 너무 많지만 않으면 되는듯
