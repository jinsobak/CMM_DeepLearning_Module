머신러닝 기법 중 RandomForest 모델 학습 및 예측 데이터.   
각 데이터에 대해 학습 후 1번씩 예측을 진행.   
각 데이터는 판정이 없는 데이터를 제외한 데이터.
===

<details>
<summary>

파일명:   
data_mv_sv_dv_ut_lt_hd_no_NTC   
(판정을 제외한 모든 항목을 사용)
---
</summary>
 
 * Test Accuracy: 0.8857142857142857   
   Test Loss: 0.2762097629786714   
   
   Confusion Matrix:
   /|Positive|Nagative|
   |:---:|:---:|:---:|
   Positive|30|16|
   Nagative|0|94|
   
   Accuracy:  0.8857142857142857   
   Log Loss:  0.2762097629786714   
   Precision:  0.8545454545454545   
   Recall:  1.0   
   F1-Score:  0.9215686274509803   
   ROC-AUC:  0.9619565217391304   
</details>

<details>
<summary>

파일명:   
data_jd_hd_no_NTC   
(판정 항목 만을 사용)
---
</summary>
 
 * Test Accuracy: 0.8928571428571429      
   Test Loss: 0.28109364105601903     
   
   Confusion Matrix:
   /|Positive|Nagative|
   |:---:|:---:|:---:|
   Positive|34|12|
   Nagative|3|91|
    
   Precision:  0.883495145631068   
   Recall:  0.9680851063829787   
   F1-Score:  0.9238578680203046   
   ROC-AUC:  0.9480804810360777   
</details>

<details>
<summary>
 
파일명:   
data_jd_hd_delete_material_no_NTC   
(판정 항목 만을 사용하되 소재라는 문자열이 들어간 요소를 삭제)
---
</summary>
 
 * Test Accuracy: 0.8857142857142857      
   Test Loss: 0.7412055843669417     
   
   Confusion Matrix:
   /|Positive|Nagative|
   |:---:|:---:|:---:|
   Positive|33|13|
   Nagative|3|91|
     
   Precision:  0.875    
   Recall:  0.9680851063829787   
   F1-Score:  0.9191919191919192   
   ROC-AUC:  0.9361702127659575      
</details>

이후 PCA진행 데이터   
(데이터 설명, 컴포넌트 개수, explained variance ratio)
---

<details>
<summary>
 
파일명:   
data_mv_sv_dv_ut_lt_hd_no_NTC_pca_component_4   
(모든 항목을 사용한 데이터, 컴포넌트 4개, 49%)
---
</summary>
 
 * Test Accuracy: 0.75      
   Test Loss: 0.4742818801426057     
   
   Confusion Matrix:
   /|Positive|Nagative|
   |:---:|:---:|:---:|
   Positive|19|27|
   Nagative|8|86|
     
   Precision:  0.7610619469026548    
   Recall:  0.9148936170212766   
   F1-Score:  0.8309178743961353   
   ROC-AUC:  0.8457446808510638      
</details>

<details>
<summary>

파일명:   
data_mv_sv_dv_ut_lt_hd_no_NTC_pca_component_26   
(모든 항목을 사용한 데이터, 컴포넌트 26개, 95%)
---
</summary>
 
 * Test Accuracy: 0.8142857142857143      
   Test Loss: 0.40197761284598305     
   
   Confusion Matrix:
   /|Positive|Nagative|
   |:---:|:---:|:---:|
   Positive|22|24|
   Nagative|2|92|
     
   Precision:  0.7931034482758621    
   Recall:  0.9787234042553191   
   F1-Score:  0.8761904761904762   
   ROC-AUC:  0.8919981498612396      
</details>

<details>
<summary>

파일명:   
data_jd_hd_no_NTC_pca_component_4   
(판정 항목 만을 사용, 컴포넌트 4개, 28%)
---
</summary>
 
 * Test Accuracy: 0.8357142857142857      
   Test Loss: 0.8324459700826609     
   
   Confusion Matrix:
   /|Positive|Nagative|
   |:---:|:---:|:---:|
   Positive|32|14|
   Nagative|9|85|
     
   Precision:  0.8585858585858586    
   Recall:  0.9042553191489362   
   F1-Score:  0.8808290155440415   
   ROC-AUC:  0.8891073080481036      
</details>

<details>
<summary>

파일명:   
data_jd_hd_no_NTC_pca_component_26   
(판정 항목 만을 사용, 컴포넌트 26개, 95%)
---
</summary>
 
 * Test Accuracy: 0.8714285714285714      
   Test Loss: 0.3032147408763927     
   
   Confusion Matrix:
   /|Positive|Nagative|
   |:---:|:---:|:---:|
   Positive|33|13|
   Nagative|5|89|
     
   Precision:  0.8725490196078431    
   Recall:  0.9468085106382979   
   F1-Score:  0.9081632653061225   
   ROC-AUC:  0.9405642923219241      
</details>

<details>
<summary>

파일명:   
data_jd_hd_delete_material_no_NTC_pca_component_4   
(판정 항목 만을 사용하되 소재라는 문자열이 들어간 열 삭제,   
컴포넌트 4개, 43%)
---
</summary>
 
 * Test Accuracy: 0.8428571428571429      
   Test Loss: 0.7844264236513456     
   
   Confusion Matrix:
   /|Positive|Nagative|
   |:---:|:---:|:---:|
   Positive|30|16|
   Nagative|6|88|
     
   Precision:  0.8461538461538461    
   Recall:  0.9361702127659575   
   F1-Score:  0.8888888888888888   
   ROC-AUC:  0.9174375578168363      
</details>

<details>
<summary>

파일명:   
data_jd_hd_delete_material_no_NTC_pca_component_7   
(판정 항목 만을 사용하되 소재라는 문자열이 들어간 열 삭제,   
컴포넌트 7개, 61%)
---
</summary>
 
 * Test Accuracy: 0.8571428571428571      
   Test Loss: 0.5727501903121676     
   
   Confusion Matrix:
   /|Positive|Nagative|
   |:---:|:---:|:---:|
   Positive|31|15|
   Nagative|5|89|
     
   Precision:  0.8557692307692307    
   Recall:  0.9468085106382979   
   F1-Score:  0.898989898989899   
   ROC-AUC:  0.9179000925069382      
</details>

<details>
<summary>

파일명:   
data_jd_hd_delete_material_no_NTC_pca_component_17   
(판정 항목 만을 사용하되 소재라는 문자열이 들어간 열 삭제,   
컴포넌트 17개, 95%)
---
</summary>
 
 * Test Accuracy: 0.8571428571428571      
   Test Loss: 0.3130116237292688     
   
   Confusion Matrix:
   /|Positive|Nagative|
   |:---:|:---:|:---:|
   Positive|29|17|
   Nagative|3|91|
     
   Precision:  0.8425925925925926    
   Recall:  0.9680851063829787   
   F1-Score:  0.900990099009901   
   ROC-AUC:  0.9330481036077706      
</details>
