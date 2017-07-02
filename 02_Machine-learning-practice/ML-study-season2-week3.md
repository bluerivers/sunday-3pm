# 5/21 ML 스터디 시즌 2 3주 차

## 진행사항

* boston housing 를 따라하는 데에도 할 것이 많아서 추가적인 것을 하지 못함
* 스터디 진행 방식 및 코드 올리는 방법을 결정함


## 차주 진행사항

* Linear Regression 관련 예제 1개씩 추가로 해보기
* Classification 관련
    * https://archive.ics.uci.edu/ml/datasets/Student+Performance

## 후기

### Luca
- 아담 옵티마이져는 좋은 것
- classification은 딱 떨어지게 label을 구하는 것보다 grouping해서 구하는 것이 더 유의미한 경우가 있음. 상중하 정도만 판단해도 된다고 하면. 1~100의 점수를 classification 하는 것은 classification이라 하기에는 이상할 듯
- softmax의 한계가 보임
- 데이터가 이쁘게 되어 있어서, string을 어떻게 처리할지 student로 배울 수 있을 듯

### Gin
- no content

### Brad
- no content

### Jay
- AdamOptimizer 짱짱맨
- classification은 결과값(label)이 n차원인 learning의 special case라고 생각하면 될 거 같다.
- xor 문제랑 비슷한 이유로 wine data의 learning 결과가 60% 초반을 넘지 못하는 거 같다.
예를들어 (내가 와알못이지만) 신맛, 단맛, 쓴맛이 0~10으로 표현될 때
(신맛 2, 단맛 5, 쓴맛 3)의 와인이 상급으로 분류된다면 softmax로 분류가 가능하겠지만
(신맛 2, 단맛 5, 쓴맛 3) and (신맛 7, 단맛 8, 쓴맛 5)이 상급으로 분류된다면
softmax의 logit이 linear이기 때문에 (hypothesis는 monotonic이 되고) 둘 다 상급으로 분류할 수 없게된다.
둘 중에 더 많이 맞는 쪽으로 weight이 learning되고 (신맛 2, 단맛 5, 쓴맛 3)을 상급으로 분류하도록 weight이 결정되면
(신맛 7, 단맛 8, 쓴맛 5)를 상급으로 분류할 수 없어서 일정 accuracy를 넘을 수 없는 거 같음
- 이번 데이터가 그런건 아니지만 다음에 할 데이터로 봤을 때,
categorical data(값의 크기가 의미를 가지지 않고 분류의 의미만 가지는 data, 설문에서 봤을 때 객관식 단일응답 같은거)를
어떻게 처리할 지가 중요해 보인다.
    - 1, 2, 3, 4... 로 label하는 방법 -> 결과값과의 관계에 따라 sorting이 잘 되어있으면 모르겠지만 (이러면 이미 값의 크기에 의미를 가지는게 되겠지만)
그렇지 않은 경우 단순하게 생각해보면 예를 들어 직업 = {의사, 개발자, 선생님}인 경우
{1: 의사, 2: 개발자, 3: 선생님}으로 labeling하는 거랑 {1: 의사, 3: 개발자, 2: 선생님}으로 labeling하는 거랑 다른 결과를 낼 거라는 것부터 이미 망할 냄새가 난다.
    - data가 분류되는 개수만큼 feature를 늘려서 one-hot encoding
의사 = {0, 1}, 개발자 = {0, 1}, 선생님 = {0, 1}로 하는 방법
첫번째보다 이게 더 그럴듯하긴 한데 feature가 늘어나는 만큼 계산시간이 늘어난다고 함
    - 더 좋은 방법이 있는지 알아봐야함
