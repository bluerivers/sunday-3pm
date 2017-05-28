# linear regression에 대한 구현 모음


## boston housing (2주차)

### 질문

#### Luca

- 값에서 평균 빼고 표준 분포로 나눈게 어떻게 노멀라이즈 되는가? z-score의 동작 원리가 궁금


### 각자 깨달은 점

#### Luca
- tensowflow가 어떻게 동작하는지도 중요하지만, 각 data processing 전후의 data의 shape 변화나 normerization으로 variance가 어떻게 변하는지나, library의 method signature도 익숙해져야 직접 코드를 쓸 수 있음을 배움

#### Gin

* Normalize 는 왜 하는가?
** multiple variate regression 에서는 각 변수의 값의 범위가 다르고 그 크기에 따라 label(Y)에 미치는 영향이 현저히 다를 수 있기 때문에 Normalize를 해야 한다고 나와있음([참고](https://stats.stackexchange.com/questions/29781/when-conducting-multiple-regression-when-should-you-center-your-predictor-varia))
* 진행하면서 Normalize 등 통계적 지식이 어느 정도는 있어야 한다고 느꼈음

#### brad
- 데이터를 처리하기 위한 제반 작업들이 상당히 중요하다고 생각함. 분석을 위해서 원하는 형태로 맞추어 가는 과정이 실제 tensorflow를 돌리는 과정보다 시간이 더 소요되었음.
- 이번 경우에는 데이터가 명확하게 분리되어 있지만, 데이터의 의미를 파악하고 분석에 사용하기까지 생각보다 많은 연습이 필요할 것이라 생각됨