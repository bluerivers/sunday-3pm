# 18/6/17 ML 스터디 시즌 5, 2주 차

> 참가자 : Luca, Gin, Jay, Brad

## 진행사항

* Faster RCNN review

## 차주 준비사항

* Faster RCNN 구현해보기

### 각자 깨달은 점

#### Luca
*


#### Gin
*


#### Brad
*


#### Jay
* Faster RCNN의 Region Proposal Network는 VGG같은 convolutional layer의
결과인 feature map을 input으로 받는다
* anchor box와 ground truth box의 IOU (intersection over union)를 가지고
positive, negative를 labeling해서 학습한다.
    * loss는 anchor index, index에 해당하는 앵커박스가 positive일 확률,
    예측된 bounding box coordinate를 가지고 계산
    * 당연히 negative가 훨씬 많은데 positive와 negative는 1:1 비율로 뽑음
    * loss의 coordinate이 anchor와 ground truth의 상대적인 위치로 계산되므로
    translate invariant인가? 그래서 절대적인 위치에 상관없이 positive, negative
    sample을 같은 개수만큼 뽑아도 상관없는가?
    * anchor box와 ground truth box는 이미지 위에 있는가? feature map 위에 있는가?
* 1 by 1 conv net은 차원을 줄이는 용도라고 볼 수도 있고 (차원을 미리
줄여놓고 계산하는게 계산량이 훨씬 적음), 채널에서 의미를 뽑아낸다고
볼 수도 있을 듯 (뭔가 Gin이 이런 얘기를 했는데 까먹음)