# 18/6/3 ML 스터디 시즌 5, 7주 차

> 참가자 : Luca, Gin, Jay

## 진행사항

* `Image object dection`
  * 구현 전략은 각자 논문 등을 정리하여 발표하는 형식이거나,
  * 동작하는 코드로 자신이 생각한 컨셉을 증명하는 것으로 공유한다.

## 차주 준비사항

* Object detection에 필요한 요소인 region proposal, cnn model 등을 학습해보고 naive하더라도 학습과 신경망 모델링과 결과까지의 한 사이클을 직접 구현해보기

### 각자 깨달은 점

#### Luca
* [진행하면서 기록한거](http://simp.ly/publish/DNzNf8)
* 일단 튜토리얼을 돌리면서 결과를 확인해보고 싶었음
* Tensorflow object detection API를 삽질 끝에 설치하고, model를 바꿔가며 같은 사진으로 다른 결과를 확인함
* 직접 찍거나 본인이 나온 사진으로 하니 무언가 된다는 체감이 확들었음
* 하지만 내가 모델을 구성하지도 않았고, 학습하지도 않았기에 아직 갈길이 멀음
* 각자 다른 영역을 준비해와서, 부족했다고 생각되는 부분에 대해 딱 알맞게 도움을 줄 수 있었음
  * 제이는 naive한 구현
  * gin은 object detection과 cnn model에 대한 소개
  * 나는 간단한 적용 결과


#### Gin

* R-CNN 논문 내용 공유 - [발표 자료](https://docs.google.com/presentation/d/1Ej1-4OcJ3Yz1G4noSwuKDRD_rBpFtqZ5f9DJFegmupw/edit?usp=sharing)
* 모두 실질적인 예를 돌려보고 올 것으로 예상해서 작동원리에 대해 알기 위해 Object Detection 관련 기술에 대한 것을 파악했고 그 시초가 되는 R-CNN 논문인 [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/pdf/1311.2524.pdf) 에 대한 공부와 발표를 준비했음
* 단순 예 실행보다 Object Detection에 대한 배경 및 도메인에 대한 얘기가 많아서 실질적으로 이 문제와 도메인에 대해 이해하는 데 좋았음. 물론 아직 손톱의 때만큼 아는 수준임.
* 다음 시간엔 R-CNN의 문제점을 개선하기 위한 Fast R-CNN 에 대해 공부하고 모델을 실질적으로 만들어 볼 예정임


#### Brad

#### Jay
* 나이브한 face detection 구현을 해봄
* true data = face recognition에 사용되는 250 by 250 얼굴 이미지, false data = object detection에 사용되는 이미지
* 각각 100개씩 CNN으로 학습 (트레이닝 데이터를 너무 적게 사용한거같다)
* 전처리를 함
  * gray scale로 변환
  * false data는 250 by 250이 아니라서 정사각형으로 crop하고 250 by 250으로 resize
* 학습 후 새로운 이미지에서 sliding window 방식으로 가장 두드러지는 부분을 찾아봄
* 결과가 좋지 않았음
  * 트레이닝을 너무 적게 했거나, 모델을 너무 막만들어서 일 수도 있을 듯, data noise가 너무 심한 걸지도 모르고 원인은 좀 더 봐야함
  * sliding window 방식은 가로세로 offset움직이고 window size조정하는게 너무 많아서 오래걸리는 문제도 있음
* threshold를 정하고 겹치는 영역을 잘 제거하면 여러개의 face를 찾아내는 것도 될거같긴한데 디테일이 부족함
