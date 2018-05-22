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

#### Brad

#### Jay
