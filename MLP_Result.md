MLP 모델 학습 및 예측 데이터.   
각 데이터에 대해 3번씩 학습 및 예측을 진행.   
각 데이터는 판정이 없는 데이터를 제외한 데이터.
===

<details>
<summary>

파일명:   
data_mv_sv_dv_ut_lt_hd_no_NTC   
(판정을 제외한 모든 항목을 사용)
---
</summary>

+ 시도1.   
   * Test Loss: 0.5478043556213379   
     Test Accuracy: 0.699999988079071
     
     Confusion Matrix:
     /|Positive|Nagative|
     |:---:|:---:|:---:|
     Positive|0|21|
     Nagative|0|49|
     
     Accuracy: 0.7   
     Precision: 0.7   
     Recall: 1.0   
     F1 Score: 0.8235294117647058   
   
+ 시도2
   * Test Loss: 0.4584580361843109   
     Test Accuracy: 0.7571428418159485   
     
     Confusion Matrix:
     /|Positive|Nagative|
     |:---:|:---:|:---:|
     Positive|12|9|
     Nagative|8|41|
     
     Accuracy: 0.7571428571428571   
     Precision: 0.82   
     Recall: 0.8367346938775511   
     F1 Score: 0.8282828282828283 
    
+ 시도3
   * Test Loss: 0.5873710513114929   
     Test Accuracy: 0.699999988079071   
     
     Confusion Matrix:
     /|Positive|Nagative|
     |:---:|:---:|:---:|
     Positive|0|21|
     Nagative|0|49|
     
     Accuracy: 0.7   
     Precision: 0.7   
     Recall: 1.0   
     F1 Score: 0.8235294117647058   
</details>

<details>
<summary>
    
파일명:   
data_jd_hd_no_NTC   
(판정 항목 만을 사용)
---
</summary>   

+ 시도1
   * Test Loss: 0.3098602890968323   
     Test Accuracy: 0.8714285492897034   
       
     Confusion Matrix:   
     /|Positive|Nagative|
     |:---:|:---:|:---:|
     Positive|14|7|
     Nagative|2|47|
     
     Accuracy: 0.8714285714285714    
     Precision: 0.8703703703703703   
     Recall: 0.9591836734693877   
     F1 Score: 0.912621359223301   
      
+ 시도2
   * Test Loss: 0.2470361441373825   
     Test Accuracy: 0.8714285492897034   
     
     Confusion Matrix:
     /|Positive|Nagative|
     |:---:|:---:|:---:|
     Positive|15|6|
     Nagative|3|46|
     
     Accuracy: 0.8714285714285714   
     Precision: 0.8846153846153846   
     Recall: 0.9387755102040817   
     F1 Score: 0.9108910891089109   

+ 시도3
   * Test Loss: 0.6667794585227966   
     Test Accuracy: 0.699999988079071   
     
     Confusion Matrix:
     /|Positive|Nagative|
     |:---:|:---:|:---:|
     Positive|0|21|
     Nagative|0|49|
     
     Accuracy: 0.7   
     Precision: 0.7   
     Recall: 1.0   
     F1 Score: 0.8235294117647058   
</details>

<details>
<summary>

파일명:   
data_jd_hd_delete_material_no_NTC   
(판정 항목 만을 사용하되 소재라는 문자열이 들어간 요소를 삭제)
---
</summary>

+ 시도1
   * Test Loss: 0.6681578159332275   
     Test Accuracy: 0.699999988079071   
     
     Confusion Matrix:   
     /|Positive|Nagative|
     |:---:|:---:|:---:|
     Positive|0|21|
     Nagative|0|49|
     
     Accuracy: 0.7   
     Precision: 0.7   
     Recall: 1.0   
     F1 Score: 0.8235294117647058   

+ 시도2
   * Test Loss: 0.22255265712738037   
     Test Accuracy: 0.8999999761581421   
     
     Confusion Matrix:
     /|Positive|Nagative|
     |:---:|:---:|:---:|
     Positive|15|6|
     Nagative|1|48|
     
     Accuracy: 0.9   
     Precision: 0.8888888888888888   
     Recall: 0.9795918367346939   
     F1 Score: 0.9320388349514563

+ 시도3
   * Test Loss: 0.20000457763671875   
     Test Accuracy: 0.9428571462631226   
     
     Confusion Matrix:
     /|Positive|Nagative|
     |:---:|:---:|:---:|
     Positive|18|3|
     Nagative|1|48|
     
     Accuracy: 0.9428571428571428   
     Precision: 0.9411764705882353   
     Recall: 0.9795918367346939   
     F1 Score: 0.96   
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
   
+ 시도1
   * Test Loss: 0.2921641767024994   
     Test Accuracy: 0.8714285492897034   
     
     Confusion Matrix:
     /|Positive|Nagative|
     |:---:|:---:|:---:|
     Positive|15|6|
     Nagative|3|46|
     
     Accuracy: 0.8714285714285714   
     Precision: 0.8846153846153846   
     Recall: 0.9387755102040817   
     F1 Score: 0.9108910891089109   

+ 시도2
   * Test Loss: 0.3355987071990967   
     Test Accuracy: 0.8285714387893677   
     
     Confusion Matrix:
     /|Positive|Nagative|
     |:---:|:---:|:---:|
     Positive|11|10|
     Nagative|2|47|
  
     Accuracy: 0.8285714285714286   
     Precision: 0.8245614035087719   
     Recall: 0.9591836734693877   
     F1 Score: 0.8867924528301887   

+ 시도3
   * Test Loss: 0.3451468050479889   
     Test Accuracy: 0.8714285492897034   
     
     Confusion Matrix:
     /|Positive|Nagative|
     |:---:|:---:|:---:|
     Positive|13|8|
     Nagative|1|48|
  
     Accuracy: 0.8714285714285714   
     Precision: 0.8571428571428571   
     Recall: 0.9795918367346939   
     F1 Score: 0.9142857142857143   
</details>

파일명:   
data_mv_sv_dv_ut_lt_hd_no_NTC_pca_component_26   
(모든 항목을 사용한 데이터, 컴포넌트 26개, 95%)
---
<details>
    <summary>시도1</summary>

    Test Loss: 0.645046055316925
    Test Accuracy: 0.699999988079071
    
    Confusion Matrix:
    [[ 0 21]
    [ 0 49]]
    Accuracy: 0.7
    Precision: 0.7
    Recall: 1.0
    F1 Score: 0.8235294117647058
</details>
<details>
    <summary>시도2</summary>

    Test Loss: 0.5980672836303711
    Test Accuracy: 0.699999988079071
    
    Confusion Matrix:
    [[ 0 21]
    [ 0 49]]
    Accuracy: 0.7
    Precision: 0.7
    Recall: 1.0
    F1 Score: 0.8235294117647058
</details>
<details>
    <summary>시도3</summary>

    Test Loss: 0.5498936176300049
    Test Accuracy: 0.699999988079071
    
    Confusion Matrix:
    [[ 0 21]
    [ 0 49]]
    Accuracy: 0.7
    Precision: 0.7
    Recall: 1.0
    F1 Score: 0.8235294117647058
</details>
    
파일명:   
data_jd_hd_no_NTC_pca_component_4   
(판정 항목 만을 사용, 컴포넌트 4개, 28%)
---
<details>
    <summary>시도1</summary>

    Test Loss: 0.2770636975765228
    Test Accuracy: 0.8999999761581421
    
    Confusion Matrix:
    [[17  4]
    [ 3 46]]
    Accuracy: 0.9
    Precision: 0.92
    Recall: 0.9387755102040817
    F1 Score: 0.9292929292929293
</details> 
<details>
    <summary>시도2</summary>

    Test Loss: 0.3183031380176544
    Test Accuracy: 0.8999999761581421
    
    Confusion Matrix:
    [[15  6]
    [ 1 48]]
    Accuracy: 0.9
    Precision: 0.8888888888888888
    Recall: 0.9795918367346939
    F1 Score: 0.9320388349514563
</details>
<details>
    <summary>시도3</summary>

    Test Loss: 0.303946852684021
    Test Accuracy: 0.9142857193946838
    
    Confusion Matrix:
    [[18  3]
    [ 3 46]]
    Accuracy: 0.9142857142857143
    Precision: 0.9387755102040817
    Recall: 0.9387755102040817
    F1 Score: 0.9387755102040817
</details>

파일명:   
data_jd_hd_no_NTC_pca_component_26   
(판정 항목 만을 사용, 컴포넌트 26개, 95%)
---
<details>
    <summary>시도1</summary>

    Test Loss: 0.6786163449287415
    Test Accuracy: 0.699999988079071
    
    Confusion Matrix:
    [[ 3 18]
    [ 3 46]]
    Accuracy: 0.7
    Precision: 0.71875
    Recall: 0.9387755102040817
    F1 Score: 0.8141592920353983
</details>
<details>
    <summary>시도2</summary>

    Test Loss: 0.24743257462978363
    Test Accuracy: 0.8714285492897034
    
    Confusion Matrix:
    [[15  6]
    [ 3 46]]
    Accuracy: 0.8714285714285714
    Precision: 0.8846153846153846
    Recall: 0.9387755102040817
    F1 Score: 0.9108910891089109
</details>
<details>
    <summary>시도3</summary>

    Test Loss: 0.52480149269104
    Test Accuracy: 0.7142857313156128
    
    Confusion Matrix:
    [[ 1 20]
    [ 0 49]]
    Accuracy: 0.7142857142857143
    Precision: 0.7101449275362319
    Recall: 1.0
    F1 Score: 0.8305084745762712
</details>

파일명:   
data_jd_hd_delete_material_no_NTC_pca_component_4   
(판정 항목 만을 사용하되 소재라는 문자열이 들어간 열 삭제,   
컴포넌트 4개, 43%)
---
<details>
    <summary>시도1</summary>

    Test Loss: 0.14326536655426025
    Test Accuracy: 0.9428571462631226
    
    Confusion Matrix:
    [[17  4]
    [ 0 49]]
    Accuracy: 0.9428571428571428
    Precision: 0.9245283018867925
    Recall: 1.0
    F1 Score: 0.9607843137254902
</details>
<details>
    <summary>시도2</summary>

    Test Loss: 0.16238410770893097
    Test Accuracy: 0.9142857193946838
    
    Confusion Matrix:
    [[17  4]
    [ 2 47]]
    Accuracy: 0.9142857142857143
    Precision: 0.9215686274509803
    Recall: 0.9591836734693877
    F1 Score: 0.94
</details>
<details>
    <summary>시도3</summary>

    Test Loss: 0.14559824764728546
    Test Accuracy: 0.9142857193946838
    
    Confusion Matrix:
    [[16  5]
    [ 1 48]]
    Accuracy: 0.9142857142857143
    Precision: 0.9056603773584906
    Recall: 0.9795918367346939
    F1 Score: 0.9411764705882353
</details>

파일명:   
data_jd_hd_delete_material_no_NTC_pca_component_7   
(판정 항목 만을 사용하되 소재라는 문자열이 들어간 열 삭제,   
컴포넌트 7개, 61%)
---
<details>
    <summary>시도1</summary>

    Test Loss: 0.26741448044776917
    Test Accuracy: 0.9142857193946838
    
    Confusion Matrix:
    [[16  5]
    [ 1 48]]
    Accuracy: 0.9142857142857143
    Precision: 0.9056603773584906
    Recall: 0.9795918367346939
    F1 Score: 0.9411764705882353
</details>
<details>
    <summary>시도2</summary>

    Test Loss: 0.20400142669677734
    Test Accuracy: 0.9571428298950195
    
    Confusion Matrix:
    [[18  3]
    [ 0 49]]
    Accuracy: 0.9571428571428572
    Precision: 0.9423076923076923
    Recall: 1.0
    F1 Score: 0.9702970297029703
</details>
<details>
    <summary>시도3</summary>

    Test Loss: 0.15940316021442413
    Test Accuracy: 0.9285714030265808
    
    Confusion Matrix:
    [[18  3]
    [ 2 47]]
    Accuracy: 0.9285714285714286
    Precision: 0.94
    Recall: 0.9591836734693877
    F1 Score: 0.9494949494949495
</details>

파일명   
data_jd_hd_delete_material_no_NTC_pca_component_17   
(판정 항목 만을 사용하되 소재라는 문자열이 들어간 열 삭제,   
컴포넌트 17개, 95%)
---
<details>
    <summary>시도1</summary>

    Test Loss: 0.2762243449687958
    Test Accuracy: 0.9142857193946838
    
    Confusion Matrix:
    [[15  6]
    [ 0 49]]
    Accuracy: 0.9142857142857143
    Precision: 0.8909090909090909
    Recall: 1.0
    F1 Score: 0.9423076923076923
</details>
<details>
    <summary>시도2</summary>

    Test Loss: 0.1931239515542984
    Test Accuracy: 0.9428571462631226
    
    Confusion Matrix:
    [[18  3]
    [ 1 48]]
    Accuracy: 0.9428571428571428
    Precision: 0.9411764705882353
    Recall: 0.9795918367346939
    F1 Score: 0.96
</details> 
<details>
    <summary>시도3</summary>

    Test Loss: 0.654637336730957
    Test Accuracy: 0.699999988079071
    
    Confusion Matrix:
    [[ 0 21]
    [ 0 49]]
    Accuracy: 0.7
    Precision: 0.7
    Recall: 1.0
    F1 Score: 0.8235294117647058
</details>

