# 18/4/8 ML 스터디 시즌 4 3주 차

> 참가자 : Jay, Luca

## 진행사항

* [집값 예측](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)의 풀이를 공유
    * Luca: Lasso
    * Jay: NN
* Lasso, Ridge regression은 L1, L2 regulization 이다.
    * https://brunch.co.kr/@itschloe1/9
    * https://brunch.co.kr/@itschloe1/11
    * http://freesearch.pe.kr/archives/4473
    * http://aikorea.org/cs231n/neural-networks-2-kr/#reg
    * https://datascienceschool.net/view-notebook/83d5e4fff7d64cb2aecfd7e42e1ece5e/
* MSE와 R-square가 왜 따로 봐야하는지? 각각의 필요성에 대한 의문을 품음
* Adjusted R-square가 필요한 이유가 feature가 증가함에 따라 R-square가 무조건 증가하거나 변하지 않기 때문이라고 함
    * 왜 그럴까 의문을 제시하였고, Jay가 Curse of Dimensionality(차원의 저주)을 알려줌
        * http://norman3.github.io/prml/docs/chapter01/4.html
            * >그림을 잘 보면 차원이 증가할수록(D가 커질수록) e 값이 작더라도 원래 볼륨 크기와 근접하게 됨을 알 수 있다. 이걸 다른 관점에서 이야기하자면 차원이 증가할수록 전체 볼륨 크기의 대부분은 표면에 위치하게 된다는 것이다.
            * 이를 이해하기가 어려웠는데, 차원이 커질 수록 1의 부피와 1-e의 부피차가 커지고
            * 각 축의 데이터들이 잘 분산되어 있더라도 중간에 밀집하는 것보다 껍데기(1-e, 1의 사이)에 분포하는게 많아짐으로 대략(?) 이해함
        * http://www.visiondummy.com/2014/04/curse-dimensionality-affect-classification/
        * https://stats.stackexchange.com/questions/99171/why-is-euclidean-distance-not-a-good-metric-in-high-dimensions
        * https://www.reddit.com/r/statistics/comments/33bu59/why_does_r2_always_increase_when_adding_new/        
* Heteroscedasticity(이분산성) : non-constant variance of [residual](https://en.wikipedia.org/wiki/Residual_(numerical_analysis))

## 차주 준비사항

* [영화 평가](https://www.kaggle.com/c/word2vec-nlp-tutorial)을 각자 풀어옴

### 각자 깨달은 점

#### Luca
* get_dummy로 non-numeric feature를 one-hot으로 바꾸는데, test랑 train dataset에서 각자 다른 데이터를 들고 있기 때문에 서로 컬럼을 맞춰줘야 했음. train 기준으로 column difference를 맞췄음
    * 그런데 다른 사람의 kaggle kernel을 보니 train과 test feature를 합쳐서 하나의 dataframe으로 만듬
    * 그리고 nan value는 전체의 평균으로 채워서 무조건 row를 날리는 것보다 더 합당해 보이는 접근임
* sale price(집값)에 log를 취하여 학습시키고 예측값에 제곱을 하는게 mse가 0.1..., 0.4...로 큰 차이를 보였음 (lasso 기준)
    * 왜 이렇게까지 차이가 나오는지 궁금함
    * skew가 높은 feature에도 log를 취했는데 이는 큰 효과는 없었음
* elastic net이 lasso, ridge를 섞은거라고 하는데 큰 차이는 없지만 lasso만 쓴 것보다 안좋은 결과가 나옴
* lasso의 feature selection 기준은 무엇인지 궁금함. correlation을 기준이라면 이는 어떻게 구하는지도 알아보면 좋을 듯함
* NN이 만능은 아니다. 데이터의 경향을 보고 방법을 선택할 수 있는 눈을 길러야겠다.
* Jay의 NN 코드가 궁금함. 아무리해도 0.2의 벽을 못넘었었는데, 내가 했던 NN과 차이를 알아보고 싶음

#### Gin


#### Brad


#### Jay