# 18/5/6 ML 스터디 시즌 4, 5주 차

> 참가자 : Luca, Brad, Gin

## 진행사항

* [영화 평가 한국](https://github.com/e9t/nsmc/)를 풀어온 소감 공유 및 서로 질문과 토론
  * 참고 - https://www.lucypark.kr/docs/2015-pyconkr/#1

## 차주 준비사항

* [영화 평가 한국](https://github.com/e9t/nsmc/)을 풀은 기존 방법을 더 개선해보거나, 주제를 확장해보기
  * 최근 영화 리뷰를 크롤링: 인피니티 워 리뷰에 적용해보기
  * 어플리케이션으로 확장: 지속 가능한 무언가를 만들어보는게 어떤가 싶음
  * (Gin idea) 리뷰가 아닌, 출연진, 감독, 스태프의 정보를 활용하여 긍부정을 알아보기

### 각자 깨달은 점

#### Luca

* (진행 중 이슈) Windows 10, Anaconda에서 konlpy 설치 중, `jupyter notebook에서 konlpy 실행 시, NameError: name 'jpype' is not defined 발생합니다ㅜ #129` [Issue](https://github.com/konlpy/konlpy/issues/129)
* 단순히 따라만 해서, 아직 word2vec과 doc2vec의 컨셉 정도만 알고 있지 구현에서의 차이를 알아봐야함
  * doc2vec의 distribued memory를 알아봐야겠다.
  * word2vec보다 doc2vec이 항상 좋은건가? 아니면 목적에 따라 다를까? 모델을 더 깊이 파보자
* 생각보다 konlpy의 stopword가 잘 적용된다. 만들어주신 분들께 감사드립니다.
* most_similar의 결과가 이상하다. 학습이 잘못되었는지, input이 잘못되었는지 대부분 0.99의 유사도를 나타냄
  * deprecated된 doc2vec init, train 코드를 바꾸면서 변경 또는 반영되지 않은 옵션이 있는 것 같다.
  * http://daewonyoon.tistory.com/240
  * 이후에, 차근차근 다시 옵션을 적용해보니 잘됨. 어떤 요소가 문제였는지는 확인이 필요함
* konlpy를 사용한 것은 앞단의 전처리만 바뀌었지 무언가 더 배웠다는 느낌이 부족했다.
  * 그 이유는 konlpy를 가져다가 사용만 했지 그 근간을 알아보지 않았다는 것과(국어 공부.. 형태소를 알아보는게 이 스터디의 방향성에 맞는가)
  * 이번에도 예제를 따라하기만 바빴지, 개인적으로 기존 지식을 활용하여 확장해보지 않음
* 따라서 다음 시간까지는 스터디의 방향성에 대해서도 다시금 제고 해보아야겠다.
  * 논문 보면서 깊이 있는 학습을 해야할까?
  * 그렇다고 모두 다 같은 주제를 파는건 보통 무난한 주제 or 어려워 동기부여가 안될 수도 있다.
  * 이제 어느정도 각자 감은 잡았으니, 각자 개인 주제를 확장해가며 깊이를 넓이거나 ML을 서비스 레벨로 확장시키는 노하우를 배우는게 어떨까?
  * 라는 이야기를 어느정도 나누었음
  * 그래도 NPL를 어정쩡하게 끝내면 안되기에 다음 시간까지는 기존에 한 것을 발전시켜보기로 함

#### Gin

#### Brad

#### Jay