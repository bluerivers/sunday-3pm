# Curriculum

우선 Deep Learning을 바로 이해하고 공부하기에는 무리가 있고, 기존 Machine Learning 의 기법을 효율적으로 적용할 수 있는 분야도 있기 때문에 이를 기반으로 쌓기 위한 스터디를 진행하기로 했다. 가장 좋은 것은 Andrew Ng 교수의 강의지만 언어적인 장벽이 있고 용어 등이 익숙하지 않기 때문에 이를 기반으로 한 국문 강의[1] 를 보면서 기본을 다져보기로 한다. (나중에 알게 됐지만 번역 자막이 있다고 하니 Ng 교수의 명강의를 들으려면 분은 참고)


## Season 1 - 딥러닝의 기본

### Week 1

* 머신러닝의 개념과 용어
* Linear Regression 의 개념
* Linear Regression cost함수 최소화


### Week 2

* 여러개의 입력(feature)의 Linear Regression
* Logistic (Regression) Classification, Hypothesis 함수 소개, Cost 함수 소개


### Week 3

* Softmax Regression (Multinomial Logistic Regression)
    * Multinomial 개념 소개
    * Cost 함수 소개 비디오
    * TensorFlow에서의 구현 비디오
* ML의 실용과 몇가지 팁 슬라이드
    * 학습 rate, Overfitting, 그리고 일반화 (Regularization)
    * Training/Testing 데이타 셋
    * TensorFlow에서의 구현 (학습 rate, training/test 셋으로 성능평가)


### Week 4

* 딥러닝의 기본 개념과, 문제, 그리고 해결
    * 딥러닝의 기본 개념: 시작과 XOR 문제
    * 딥러닝의 기본 개념2: Back-propagation 과 2006/2007 '딥'의 출현
* Neural Network 1: XOR 문제와 학습방법, Backpropagation (1986 breakthrough)
    * XOR 문제 딥러닝으로 풀기
    * 특별편: 10분안에 미분 정리하기
    * 딥넷트웍 학습 시키기 (backpropagation)
    * 실습1: XOR을 위한 텐스플로우 딥넷트웍
    * 실습2: Tensor Board로 딥네트웍 들여다보기


### Week 5
* Neural Network 2: ReLU and 초기값 정하기 (2006/2007 breakthrough)
* XSigmoid 보다 ReLU가 더 좋아
* Weight 초기화 잘해보자
* Dropout 과 앙상블
* 레고처럼 넷트웍 모듈을 마음껏 쌓아 보자
* 실습: 딥러닝으로 MNIST 98%이상 해보기


### 실습 or 과제

* https://www.datacamp.com/community/open-courses/kaggle-python-tutorial-on-machine-learning#gs.gdFOjLg

### Data classification

* http://www.mmds.org

## Reference

1. [모두를 위한 머신러닝과 딥러닝의 강의](http://hunkim.github.io/ml/)
1. [Stanford Machine Learning Notes](http://www.holehouse.org/mlclass/)
1. [Tensorflow 사용법 번역](https://tensorflowkorea.gitbooks.io/tensorflow-kr/)
1. [텐서플로우 블로그](https://tensorflow.blog)
1. [강의노트 - 모두를 위한 머신러닝과 딥러닝의 강의](http://pythonkim.tistory.com/category/머신러닝_김성훈교수님)
1. [초짜 대학원생의 입장에서 정리한 NIPS 2016 tutorial: Nuts and bolts of building AI applications using Deep Learning by Andrew Ng](http://jaejunyoo.blogspot.com/2017/03/kr-nips-2016-tutorial-summary-nuts-and-bolts-of-building-AI-AndrewNg.html)
