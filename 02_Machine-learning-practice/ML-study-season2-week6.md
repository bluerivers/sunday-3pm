# 7/02 ML 스터디 시즌 2 6주 차

## 진행사항
deep neural net을 또 해보았다

## 차주 진행사항

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

### Gin

### Brad

### Jay
* initial 값에 따라 결과가 꽤 달라지므로 initialization을 잘 해야 할 거 같다.
* feature를 one-hot encoding 해서 해보기
* regularization 같은 걸 적용해봐야겠다.
* optimal (sub-optimal?)은 실험해보면서 찾아야하지만 보통 layer 3개면 일반적은 문제를 푸는데 문제가 없다고 한다.
neuron의 개수도 input neuron의 개수에 비해 너무 많지만 않으면 되는듯
