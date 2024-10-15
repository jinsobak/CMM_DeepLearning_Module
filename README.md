# CMM_DeepLearning_Module


CMM (3차원 측정기)을 통해 특정 부품의 각도, 면적, 길이 등을 측정하여 얻은 데이터에서
이상치(에러)를 탐지해 부품의 불량 여부를 판단하는 딥러닝 모듈 개발

기존 불량 검출 시스템:
CMM을 통해 제품의 각종 수치를 측정 -> 측정실 작업자가 측정된 데이터를 확인 후 정상, 불량, 보류를 판별
-> 보류로 판별된  데이터의 경우 책임자에게 전달 -> 최종 정상, 불량 판별

이러한 기존 불량 검출 시스템의 불편함을 개선하고자 딥러닝을 이용한 이상치 탐지 모델을 개발하는 프로젝트입니다.

# 모델 및 데이터에 따른 정확도 구분


### [MLP모델(MLP_Result.md)](model_results/MLP_Result.md)

### [랜덤포레스트모델(RF_Result.md)](model_results/RF_Result.md)

# 각 모델 결과에 대한 시각화


**visualization.md**
