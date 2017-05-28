# linear regression에 대한 구현 모음


## boston housing (2주차)

### 질문

#### Luca

- 값에서 평균 빼고 표준 분포로 나눈게 어떻게 노멀라이즈 되는가? z-score의 동작 원리가 궁금

#### Gin

- test set으로 돌렸을 때 예측값과 테스트 값의 오차의 평균이 약 22 정도 나오는데 이것이 어떤 의미가 있는가?


### 각자 깨달은 점

#### Luca
- tensowflow가 어떻게 동작하는지도 중요하지만, 각 data processing 전후의 data의 shape 변화나 normerization으로 variance가 어떻게 변하는지나, library의 method signature도 익숙해져야 직접 코드를 쓸 수 있음을 배움

#### Gin

* Normalize 는 왜 하는가?
** multiple variate regression 에서는 각 변수의 값의 범위가 다르고 그 크기에 따라 label(Y)에 미치는 영향이 현저히 다를 수 있기 때문에 Normalize를 해야 한다고 나와있음([참고](https://stats.stackexchange.com/questions/29781/when-conducting-multiple-regression-when-should-you-center-your-predictor-varia))
* 진행하면서 Normalize 등 통계적 지식이 어느 정도는 있어야 한다고 느꼈음
* W 값을 보면서 각 field 이 Y에 미치는 영향, 즉 상관관계를 분석하니 좀더 명확히 느껴졌음

#### brad
- 데이터를 처리하기 위한 제반 작업들이 상당히 중요하다고 생각함. 분석을 위해서 원하는 형태로 맞추어 가는 과정이 실제 tensorflow를 돌리는 과정보다 시간이 더 소요되었음.
- 이번 경우에는 데이터가 명확하게 분리되어 있지만, 데이터의 의미를 파악하고 분석에 사용하기까지 생각보다 많은 연습이 필요할 것이라 생각됨

#### Jay
normalize의 중요성
- 이번 case에서는 learning rate을 매우 낮게 잡으면 발산하는 걸 막을 수 있었지만 빨리 수렴하지 않아서 문제, epoch을
충분히 크게 했을 때 cost가 원하는 만큼까지 줄어들지도 의문
- cost function의 gradient를 보면 각 feature만 자신의 gradient에 영향을 주는게 아니고 전체 feature가 영향을 주기 때문에
scale이 큰 feature가 영향력이 클 수 밖에 없음 -> 따라서 normalization 해야 한다.
- 여기서 하나 의문은 그냥 raw data 쓰는거보다는 normalize data 쓰는게 나은건 맞는거 같은데, 사실 feature 별로 영향력이
같다고 놓는 것도 좋은 건지 모르겠음. (같은 scale에 같은 learning rate을 쓰니까)
- GradientDescentOptimizer랑 달리 AdamOptimizer는 feature 별로 다른 learning rate을 가진다고 하는거 같은데 알아봐야됨
