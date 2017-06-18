# 5/21 ML 스터디 시즌 2 4주 차

## 진행사항

* Wine quality 분류에서 data processing에 대한 부족함을 느끼고, 마침 optional하게 준비된 student dataset을 진행함
* student dataset은 마침 string feature 들이 있어서, numeric feature만 다루었던 부족함을 채우기에 좋은 주제
* 각자 해온 결과, accurray가 20%을 안정적으로 못넘겼었음
  * 이유는 label인 0~20 범위를 갖는 점수 그대로 예측을 하려 했기 때문
  * domain을 이해하고 목적을 갖고 분석을 한다면 clustering을 해도 문제가 없기에 상중하 또는 4~7개의 category로 묶음 (정확한 점수보단, 점수의 경향성만 봐도 된다고 생각함)
* Regularization, Label clustering, outlier drop 등으로 70~90% 까지 정확도를 올림

## 각자 느낀점

### Luca
- plot으로 각 feature의 특성을 더 깊게 생각해보는 것이 필요하다고 느낌
- 정확한 점수 예측이 굳이 ML로 해야하는 목적이 아님을 깨달음. domain data와 이것으로 얻고자 하는 목적에 대해서 명확히 하는게 중요하다고 깨달음
- 여러 data processing 기술(?)을 이번에 많이 사용하여, 그만큼 효과를 봐서 좋았음

### Gin


### Jay


## 차주 진행사항
