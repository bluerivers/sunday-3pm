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
