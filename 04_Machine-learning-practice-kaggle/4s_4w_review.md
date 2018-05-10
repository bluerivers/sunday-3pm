# 18/4/22 ML 스터디 시즌 4, 4주 차

> 참가자 : Jay, Luca, Brad, Gin

## 진행사항

* [영화 평가](https://www.kaggle.com/c/word2vec-nlp-tutorial)를 풀어온 소감 공유 및 서로 질문과 토론

## 차주 준비사항

* [영화 평가 한국](https://github.com/e9t/nsmc/)을 풀어본다.
  * 참고 - https://www.lucypark.kr/docs/2015-pyconkr/#1

### 각자 깨달은 점

#### Luca

* 예전에 간단히 word2vec을 접하고, 위키를 긁어와 unsimilar word를 리스트에서 골라내는 정도를 테스트 해본적이 있었다.
  * 그때는 학습 후 데이터 형태도 몰랐고, 이걸 어떻게 활용해서 문장과 단락에 적용할지 몰랐는데, 튜토리얼을 차근차근 진행하면서 아래와 같은 방식들을 배웠다.
    * word를 어떻게 학습할 것인가?
    * sentence를 어떻게 학습할 것인가?
      * 여기서는 stop word를 제거하여 마침표를 이용하여 sentence를 구분할련지 모르겠다. 아니면 마침표를 안없애면 word2vec 과정중에 자연스래 학습이 될까? 연구자료가 있을 듯한데 더 팔로우업 해보면 좋은 토픽으로 생각됨
    * paragraph를 어떻게 학습할 것인가?
    * 평가 모델을 어떻게 만들것인가?
* bag of words
  * (단순) 문단 전체를 word로 쪼개어, word count vector를 만듬
  * 단어의 순서, context가 반영되지 않는다는 단점이 있음
  * 단어가 많아질 수록 차원이 높아져, 차원의 저주에 걸릴지도
  * 예제에서는 sklearn의 CountVectorizer를 사용했는데, max_feature라는 param을 제공하는 것을 보니 most words에 대해서만 count하는 듯 보인다. stop_words param도 있어서, 전처리가 꼭 필요한 것은 아닌 것으로 보이나 nltk.corpus.stopwords와는 차이가 있을 것으로 보임
* word2vec
  * window size로 한 word 기준으로 전후 word를 예측할 크기를 조정 가능
    * bag of word와 달리, 단어 순서로 인한 context를 보존할 수 있음
  * word2vec model을 train과 test 데이터 모두 넣어 학습 가능
    * unsupervised? 더 많은 데이터 활용 가능
  * word2vec model 학습 시, sentence 마다 학습
    * paragraph 간의 context는 보장 못함(?)
  * 평가를 위한 리뷰의 vector를 만들기
    * 한 리뷰를 문장이 아닌 문단의 wordlist로 만듬
    * word2vec은 num_features = 300, min_word_count = 40 의 옵션을 사용함
    * 이로서, 차원이 너무 많아져 복잡해지는 문제를 줄일 수 있고, 모든 단어를 판단하지 않아도 됨
  * 문장 혹은 문단을 word2vec을 활용하여 vector로 만드는 두가지 방법이 있는데, Vector Averaging과 Bag of centroid가 있음
  * Vector Averaging: 각 word에 대한 vector를 더하여 해당하는 단어 수 만큼 나눔
    * 따라서 문단의 wordlist를 word2vec을 이용한 vector로 만들 때는, 학습한 word2vec model에 포함된 단어인지 확인하고
    * shape가 (len(reviews), num_features)인 vector에 계속 더 함
    * word2vec model에 해당하는 단어 만큼, 위 vector를 나눔
  * Bag of centroid는 한정된 feature로 여러 단어들을 표현하기 때문에 어느정도 군집이 생김
    * 이를 활용하여, k-mean cluster로 각 단어들의 군집 index를 bag of word 처럼 counting 한다. 
* 영어는 stopword 골라내기가 편할거 같다.
  * context에 stopword가 영향을 줄 수도 있고, tf-idf로 중요도는 걸러지기 때문에 꼭 빼야하는 건 아님
* 결과 : 모두 평가는 random forest로 진행함
  * Bag of centroid (Word2Vec + kmean clustering): 0.84072
  * Word2Vec average vector: 0.83120
  * Bag of words: 0.84600
  * 데이터가 긍정, 부정 반반이라면 찍어도 최소 0.5이므로 그리 높은 결과는 아닐 것이다.
* word2vec은 주어진 학습 데이터가 아닌 다른 데이터로부터 학습하여 활용할 수 있으므로, 정확도가 더 낮게 나왔다하여 안좋은건 아니다.
  * 위키백과나 뉴스를 학습하여 매우큰 모델을 만들어 어디에나 적용할 수도 있고, 특정 주제, 어휘와 문장을 사용하는 데이터를 넣어 학습하고자 하는 목표에 강점을 둘 수도 있다.
* 평가 모델을 random forest만 사용했는데, ml을 이용한 regression도 적용해보면 어떨까 싶다.
* train data를 쪼개 그 요소(단어)에 대해 feature vector를 만든다는 개념이 새로웠다. 그 과정(word2vec)에 ML 개념 들어가니 X,Y가 딱 정해진데에만 쓰이는게 아닌 것을 다시 체감함
* bag of word에서 count vector 말고 tfidf vector를 쓰면 어떤 결과가 나올까?

#### Gin

* 전주에 참여하지 못하고 준비를 많이 하지 못해서 따라가기 어려웠음
* bag of words, word2vec의 개념을 이해가 부족했음
* 얼릉 실습을 해서 따라가자!
* 실질적인 것을 만들면서 고도화하는 것도 재미있을 것 같다는 의견을 나눴음

#### Brad

#### Jay
