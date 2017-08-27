# 6/18 ML 스터디 시즌 2 4주 차

## 진행사항

* Wine quality 분류에서 data processing에 대한 부족함을 느끼고, 마침 optional하게 준비된 student dataset을 진행함
* student dataset은 마침 string feature 들이 있어서, numeric feature만 다루었던 부족함을 채우기에 좋은 주제
* 각자 해온 결과, accurray가 20%을 안정적으로 못넘겼었음
  * 이유는 label인 0~20 범위를 갖는 점수 그대로 예측을 하려 했기 때문
  * domain을 이해하고 목적을 갖고 분석을 한다면 clustering을 해도 문제가 없기에 상중하 또는 4~7개의 category로 묶음 (정확한 점수보단, 점수의 경향성만 봐도 된다고 생각함)
* Regularization, Label clustering, outlier drop 등으로 70~90% 까지 정확도를 올림

## 차주 진행사항

* 이번 주에 배운 바를 적용해서 wine quality data set 의 예측 정확도 높여보기
* Wine quality 나 Student grade data set 중 하나를 neural net으로 구현해보기


## 후기

### Luca
- plot으로 각 feature의 특성을 더 깊게 생각해보는 것이 필요하다고 느낌
- 정확한 점수 예측이 굳이 ML로 해야하는 목적이 아님을 깨달음. domain data와 이것으로 얻고자 하는 목적에 대해서 명확히 하는게 중요하다고 깨달음
- 여러 data processing 기술(?)을 이번에 많이 사용하여, 그만큼 효과를 봐서 좋았음
- feature 간의 선형 정도를 파악하여, outlier를 제거하면 더 좋은 결과를 얻을 수 있음을 배움
  - pandas plotting에서 scatter matrix를 참고 
  - https://pandas.pydata.org/pandas-docs/stable/visualization.html#visualization-scatter-matrix

### Gin
- Regularization을 적용해보니 overfitting, training 때의 accuracy와 test의 accuracy 의 차이가 나는 현상을 보정할 수 있다는 것을 봐서 좋았다.
- 다만 계수는 예측값으로 넣는거 같아서 슬펐음, 좋은 방법 없을까?
- regularization의 다른 방법은 없나? 이번에는 l2loss를 사용함
- outlier 발라내기, 데이터 간 상관 관계 보기 등 데이터를 일단 보는 것이 중요하다는 것을 느꼈음
- 도메인을 파악하고 내가 구하고자 하는 바를 명확히 하는 것이 문제를 해결해나가는데 중요하다는 것을 또 느꼈음

### Jay
* wine quality나 student data나 classification 문제보다는 regression 문제에 가까운 거 같다.
* classification으로 하려면 큰 group으로 나누는 정도로 하는 수 밖에 없는듯
* 이 data들은 regression 관점에서 봤을 때 linear function을 따르지도 않는 거 같고, classification 관점에서 봤을 때
linearly separable 하지도 않은 듯 (이거는 k-nearest neighbor로만 해봐서 확실하진 않음, SVM 같은걸로 될지도)
* non-linear인 방법으로 잘 되려나
