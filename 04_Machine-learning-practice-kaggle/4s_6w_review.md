# 18/5/22 ML 스터디 시즌 4, 6주 차

> 참가자 : Luca, Brad, Gin, Jay

## 진행사항

* [영화 평가 한국](https://github.com/e9t/nsmc/)을 풀은 기존 방법을 더 개선해보거나, 주제를 확장해보기
  * 최근 영화 리뷰를 크롤링: 인피니티 워 리뷰에 적용해보기
  * 어플리케이션으로 확장: 지속 가능한 무언가를 만들어보는게 어떤가 싶음
  * (Gin idea) 리뷰가 아닌, 출연진, 감독, 스태프의 정보를 활용하여 긍부정을 알아보기

## 차주 준비사항

*

### 각자 깨달은 점

#### Luca

* doc2vec model을 저장해두니 시간을 절약할 수 있었음
* 오랜만에 신경망 모델 짜려니 vector shape가 날 괴롭혔다.
  * input data를 batch로 넣으니, 1d array를 row가 아닌 column으로 인식하여 [1, 100] is not same [100, 1] 이런 식의 오류를 겪어 onehot을 적용함
* tensorflow binary classification 적용
  * doc2vec infer_vector로 얻은 결과(0.72)가 doc2vec kmean word(?) centroid(0.85)에 비해 안좋았다.
* 그래도 이전 방식보다는 잘나왔음
  * 최빈 단어 2000개의 exist vector + 베이지안 = 0.80
  * doc2vec infer_vector + 로지스틱 = 0.65
* 이러한 결과의 차이를 내는 방식의 차이를 아직 이해하지 못함
* 이번에도 되게하는 것에 초점을 맞추었지, 정확히 doc2vec을 kmean으로 활용한게 맞는지도 모르겠음
  * word 단위로 분석된다면 doc2vec이 아닐텐데
* doc2vec과 word2vec의 차이를 알아봐야겠다.
  * 답은 논문 읽기

#### Gin

* 입력 값을 learning을 돌릴 수 있는 상태로 형태로 저장해 놓는 것이 훨씬 빠른 iteration 에 도움이 된다. ML에 있어서 이런 iteration 의 속도는 가설-검증 사이클의 핵심일 것으로 보인다.
  * [numpy save](https://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html) 를 이용하면 결과를 저장하고 load를 통해 그 데이터를 불러올 수 있다. 저장할 때 확장자로 npy 가 붙으니 참고하길!
* doc2vec model을 만들 때 epoch을 1로 주는 실수를 했는데 오히려 예측 결과는 더 잘 나오는 것을 확인했다. train data가 결과를 예측하는데 부족하거나 epoch을 많이 돌리면 overfitting이 되는 것 아닌가 생각하게 됐다.
* 다른 모델을 도입해서 tensorflow/keras 등을 따라하기만 하니 내 것이 되지 않았다는 생각을 하게 됐다. 천천히 한 개를 잡고 꾸준히 돌려봐야 온전히 내 것이 되지 않을까 하는 생각을 했다.


#### Brad

#### Jay
