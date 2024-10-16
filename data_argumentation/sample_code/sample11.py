import numpy as np
import pandas as pd

def sample_data(num_samples=10):
    측정값_평면1_평면도 = np.round(np.random.uniform(0, 0.004, num_samples), 3)
    기준값_평면1_평면도 = np.random.choice(0.1, num_samples)
    상한공차_평면1_평면도 = np.full(num_samples, 0.1)
    하한공차_평면1_평면도 = np.full(num_samples, 0)
    편차_평면1_평면도 = 기준값_평면1_평면도 - 측정값_평면1_평면도
    
    return (측정값_평면1_평면도, 기준값_평면1_평면도, 상한공차_평면1_평면도, 하한공차_평면1_평면도, 편차_평면1_평면도)

def additional_sample_data(num_samples=10):
    측정값_원1_I_D = np.round(np.random.uniform(16.465, 16.509, num_samples), 3)
    기준값_원1_I_D = np.random.choice([16.48, 16.485], num_samples)
    상한공차_원1_I_D = np.full(num_samples, 0.03)
    하한공차_원1_I_D = np.random.choice([-0.01, 0], num_samples)
    편차_원1_I_D = 기준값_원1_I_D - 측정값_원1_I_D

    측정값_원2_I_D = np.round(np.random.uniform(16.457, 16.514, num_samples), 3)
    기준값_원2_I_D = np.round(np.random.uniform(16.48, 16.485, num_samples), 3)
    상한공차_원2_I_D = np.full(num_samples, 0.03)
    하한공차_원2_I_D = np.round(np.random.uniform(-0.01, 0, num_samples), 3)
    편차_원2_I_D = 기준값_원2_I_D - 측정값_원2_I_D

    측정값_원3_I_D = np.round(np.random.uniform(16.466, 16.509, num_samples), 3)
    기준값_원3_I_D = np.round(np.random.uniform(16.48, 16.485, num_samples), 3)
    상한공차_원3_I_D = np.full(num_samples, 0.03)
    하한공차_원3_I_D = np.round(np.random.uniform(-0.01, 0, num_samples), 3)
    편차_원3_I_D = 기준값_원3_I_D - 측정값_원3_I_D

    측정값_원통1_I_D = np.round(np.random.uniform(16.465, 16.51, num_samples), 3)
    기준값_원통1_I_D = np.round(np.random.uniform(16.48, 16.485, num_samples), 3)
    상한공차_원통1_I_D = np.full(num_samples, 0.03)
    하한공차_원통1_I_D = np.round(np.random.uniform(-0.01, 0, num_samples), 3)
    편차_원통1_I_D = 기준값_원통1_I_D - 측정값_원통1_I_D

    측정값_원통1_I_직각도 = np.round(np.random.uniform(0.001, 0.117, num_samples), 3)
    기준값_원통1_I_직각도 = np.full(num_samples, 0.05)
    상한공차_원통1_I_직각도 = np.full(num_samples, 0.05)
    하한공차_원통1_I_직각도 = np.full(num_samples, 0)
    편차_원통1_I_직각도 = 측정값_원통1_I_직각도 - 기준값_원통1_I_직각도

    측정값_점2_X = np.round(np.random.uniform(116.54, 116.747, num_samples), 3)
    기준값_점2_X = np.full(num_samples, 116.6)
    상한공차_점2_X = np.full(num_samples, 0.1)
    하한공차_점2_X = np.full(num_samples, 0)
    편차_점2_X = 측정값_점2_X - 기준값_점2_X

    측정값_점2_Y = np.round(np.random.uniform(-10.914, -10.894, num_samples), 3)
    기준값_점2_Y = np.full(num_samples, -10.9)
    상한공차_점2_Y = np.full(num_samples, 0.1)
    하한공차_점2_Y = np.full(num_samples, -0.1)
    편차_점2_Y = 측정값_점2_Y - 기준값_점2_Y

    측정값_각도1 = np.round(np.random.uniform(55.365, 57.486, num_samples), 3)
    기준값_각도1 = np.full(num_samples, 57)
    상한공차_각도1 = np.full(num_samples, 0.333)
    하한공차_각도1 = np.full(num_samples, -0.333)
    편차_각도1 = 측정값_각도1 - 기준값_각도1

    측정값_직선4 = np.round(np.random.uniform(26.939, 27.389, num_samples), 3)
    기준값_직선4 = np.full(num_samples, 27)
    상한공차_직선4 = np.full(num_samples, 0.5)
    하한공차_직선4 = np.full(num_samples, -0.5)
    편차_직선4 = 측정값_직선4 - 기준값_직선4

    측정값_직선5 = np.round(np.random.uniform(7.354, 7.98, num_samples), 3)
    기준값_직선5 = np.full(num_samples, 7.5)
    상한공차_직선5 = np.full(num_samples, 0.5)
    하한공차_직선5 = np.full(num_samples, -0.5)
    편차_직선5 = 기준값_직선5 - 측정값_직선5

    측정값_점3 = np.round(np.random.uniform(109.91, 110.133, num_samples), 3)
    기준값_점3 = np.full(num_samples, 110.2)
    상한공차_점3 = np.full(num_samples, 0.3)
    하한공차_점3 = np.full(num_samples, -0.3)
    편차_점3 = 기준값_점3 - 측정값_점3

    측정값_직선6 = np.round(np.random.uniform(35.397, 36.16, num_samples), 3)
    기준값_직선6 = np.full(num_samples, 35.9)
    상한공차_직선6 = np.full(num_samples, 0.5)
    하한공차_직선6 = np.full(num_samples, -0.5)
    편차_직선6 = 기준값_직선6 - 측정값_직선6

    측정값_점4_X = np.round(np.random.uniform(87.709, 88.066, num_samples), 3)
    기준값_점4_X = np.full(num_samples, 88)
    상한공차_점4_X = np.full(num_samples, 0.3)
    하한공차_점4_X = np.full(num_samples, -0.3)
    편차_점4_X = 기준값_점4_X - 측정값_점4_X

    측정값_점4_Y = np.round(np.random.uniform(-24.437, 24.21, num_samples), 3)
    기준값_점4_Y = np.full(num_samples, -24)
    상한공차_점4_Y = np.full(num_samples, 0.3)
    하한공차_점4_Y = np.full(num_samples, -0.3)
    편차_점4_Y = 기준값_점4_Y - 측정값_점4_Y

    측정값_직선7 = np.round(np.random.uniform(-23.437, 23.1, num_samples), 3)
    기준값_직선7 = np.full(num_samples, -23.1)
    상한공차_직선7 = np.full(num_samples, 0.5)
    하한공차_직선7 = np.full(num_samples, -0.5)
    편차_직선7 = 기준값_직선7 - 측정값_직선7

    측정값_직선8 = np.round(np.random.uniform(-6.278, -6.1, num_samples), 3)
    기준값_직선8 = np.full(num_samples, -6)
    상한공차_직선8 = np.full(num_samples, 0.5)
    하한공차_직선8 = np.full(num_samples, -0.5)
    편차_직선8 = 기준값_직선8 - 측정값_직선8

    측정값_점5_X = np.round(np.random.uniform(47.822, 48.103, num_samples), 3)
    기준값_점5_X = np.full(num_samples, 48)
    상한공차_점5_X = np.full(num_samples, 0.3)
    하한공차_점5_X = np.full(num_samples, -0.3)
    편차_점5_X = 기준값_점5_X - 측정값_점5_X

    측정값_점5_Y = np.round(np.random.uniform(-16.363, -16.182, num_samples), 3)
    기준값_점5_Y = np.full(num_samples, -16.1)
    상한공차_점5_Y = np.full(num_samples, 0.3)
    하한공차_점5_Y = np.full(num_samples, -0.3)
    편차_점5_Y = 기준값_점5_Y - 측정값_점5_Y

    측정값_원4_X = np.round(np.random.uniform(-0.079, 0.08, num_samples), 3)
    기준값_원4_X = np.full(num_samples, 0)
    상한공차_원4_X = np.full(num_samples, 0.2)
    하한공차_원4_X = np.full(num_samples, -0.2)
    편차_원4_X = 기준값_원4_X - 측정값_원4_X

    측정값_원4_E_소재_Y = np.round(np.random.uniform(-0.006, 0.239, num_samples), 3)
    기준값_원4_E_소재_Y = np.full(num_samples, 0)
    상한공차_원4_E_소재_Y = np.full(num_samples, 0.2)
    하한공차_원4_E_소재_Y = np.full(num_samples, -0.2)
    편차_원4_E_소재_Y = np.round(np.random.uniform(0, 0.239, num_samples), 3)

    측정값_원4_E_소재_D = np.round(np.random.uniform(25.653, 25.732, num_samples), 3)
    기준값_원4_E_소재_D = np.full(num_samples, 26.1)
    상한공차_원4_E_소재_D = np.full(num_samples, 0)
    하한공차_원4_E_소재_D = np.full(num_samples, -0.5)
    편차_원4_E_소재_D = np.round(np.random.uniform(-0.447, -0.368, num_samples), 3)

    측정값_점6_소재_X = np.round(np.random.uniform(44.059, 44.303, num_samples), 3)
    기준값_점6_소재_X = np.full(num_samples, 44.1)
    상한공차_점6_소재_X = np.full(num_samples, 0.3)
    하한공차_점6_소재_X = np.full(num_samples, -0.3)
    편차_점6_소재_X = np.round(np.random.uniform(-0.041, 0.203, num_samples), 3)

    측정값_점6_소재_Y = np.round(np.random.uniform(6.514, 6.725, num_samples), 3)
    기준값_점6_소재_Y = np.full(num_samples, 6.5)
    상한공차_점6_소재_Y = np.full(num_samples, 0.3)
    하한공차_점6_소재_Y = np.full(num_samples, -0.3)
    편차_점6_소재_Y = np.round(np.random.uniform(0.014, 0.225, num_samples), 3)

    측정값_점7_X = np.round(np.random.uniform(54.012, 54.597, num_samples), 3)
    기준값_점7_X = np.full(num_samples, 54)
    상한공차_점7_X = np.full(num_samples, 0.3)
    하한공차_점7_X = np.full(num_samples, -0.3)
    편차_점7_X = np.round(np.random.uniform(0.012, 0.597, num_samples), 3)
    
    측정값_점7_Y = np.round(np.random.uniform(5.103, 5.272, num_samples), 3)
    기준값_점7_Y = np.full(num_samples, 5.1)
    상한공차_점7_Y = np.full(num_samples, 0.3)
    하한공차_점7_Y = np.full(num_samples, -0.3)
    편차_점7_Y = np.round(np.random.uniform(0.003, 0.172, num_samples), 3)
    
    측정값_점8_X = np.round(np.random.uniform(61.442, 74.54, num_samples), 3)
    기준값_점8_X = np.full(num_samples, 77.4)
    상한공차_점8_X = np.full(num_samples, 0.3)
    하한공차_점8_X = np.full(num_samples, -0.3)
    편차_점8_X = np.round(np.random.uniform(-12.958, 0.14, num_samples), 3)

    # 직선14 데이터 생성
    측정값_직선14 = np.round(np.random.uniform(-23.755, 22.997, num_samples), 3)
    기준값_직선14 = np.full(num_samples, -23.1)
    상한공차_직선14 = np.full(num_samples, 0.333)
    하한공차_직선14 = np.full(num_samples, -0.333)
    편차_직선14 = 기준값_직선14 - 측정값_직선14
    
    # 직선16 데이터 생성
    측정값_직선16 = np.round(np.random.uniform(13.701, 15.142, num_samples), 3)
    기준값_직선16 = np.full(num_samples, 14.5)
    상한공차_직선16 = np.full(num_samples, 0.5)
    하한공차_직선16 = np.full(num_samples, -0.5)
    편차_직선16 = 기준값_직선16 - 측정값_직선16
    
    # 직선18 데이터 생성
    측정값_직선18 = np.round(np.random.uniform(-15.431, -13.271, num_samples), 3)
    기준값_직선18 = np.full(num_samples, -14.5)
    상한공차_직선18 = np.full(num_samples, 0.5)
    하한공차_직선18 = np.full(num_samples, -0.5)
    편차_직선18 = 기준값_직선18 - 측정값_직선18

    측정값_거리1 = np.round(np.random.uniform(10.287, 10.391, num_samples), 3)
    기준값_거리1 = np.full(num_samples, 10.3)
    상한공차_거리1 = np.full(num_samples, 0.07)
    하한공차_거리1 = np.full(num_samples, -0.07)
    편차_거리1 = 기준값_거리1 - 측정값_거리1

    측정값_점13 = np.round(np.random.uniform(72.801, 72.963, num_samples), 3)
    기준값_점13 = np.full(num_samples, 72.87)
    상한공차_점13 = np.full(num_samples, 0.1)
    하한공차_점13 = np.full(num_samples, 0)
    편차_점13 = 기준값_점13 - 측정값_점13

    측정값_직선19 = np.round(np.random.uniform(-15.126, -13.118, num_samples), 3)
    기준값_직선19 = np.full(num_samples, -14.5)
    상한공차_직선19 = np.full(num_samples, 0.5)
    하한공차_직선19 = np.full(num_samples, -0.5)
    편차_직선19 = 기준값_직선19 - 측정값_직선19

    측정값_직선21 = np.round(np.random.uniform(12.878, 15.083, num_samples), 3)
    기준값_직선21 = np.full(num_samples, 14.5)
    상한공차_직선21 = np.full(num_samples, 0.5)
    하한공차_직선21 = np.full(num_samples, -0.5)
    편차_직선21 = 기준값_직선21 - 측정값_직선21
 
    측정값_거리2 = np.round(np.random.uniform(10.251, 10.361, num_samples), 3)
    기준값_거리2 = np.full(num_samples, 10.3)
    상한공차_거리2 = np.full(num_samples, 0.07)
    하한공차_거리2 = np.full(num_samples, -0.07)
    편차_거리2 = 기준값_거리2 - 측정값_거리2

    측정값_점18 = np.round(np.random.uniform(72.814, 72.985, num_samples), 3)
    기준값_점18 = np.full(num_samples, 72.87)
    상한공차_점18 = np.full(num_samples, 0.1)
    하한공차_점18 = np.full(num_samples, 0)
    편차_점18 = 기준값_점18 - 측정값_점18

    측정값_점19 = np.round(np.random.uniform(-2.407, -2.292, num_samples), 3)
    기준값_점19 = np.full(num_samples, -2.3)
    상한공차_점19 = np.full(num_samples, 0.1)
    하한공차_점19 = np.full(num_samples, -0.1)
    편차_점19 = 기준값_점19 - 측정값_점19

    측정값_점20 = np.round(np.random.uniform(-2.382, -2.179, num_samples), 3)
    기준값_점20 = np.full(num_samples, -2.3)
    상한공차_점20 = np.full(num_samples, 0.1)
    하한공차_점20 = np.full(num_samples, -0.1)
    편차_점20 = 기준값_점20 - 측정값_점20

    측정값_원5 = np.round(np.random.uniform(1.898, 2.115, num_samples), 3)
    기준값_원5 = np.full(num_samples, 2)
    상한공차_원5 = np.full(num_samples, 0.1)
    하한공차_원5 = np.full(num_samples, -0.1)
    편차_원5 = 기준값_원5 - 측정값_원5

    측정값_원5_상부 = np.round(np.random.uniform(3.479, 3.576, num_samples), 3)
    기준값_원5_상부 = np.full(num_samples, 3.5)
    상한공차_원5_상부 = np.full(num_samples, 0.1)
    하한공차_원5_상부 = np.full(num_samples, -0.1)
    편차_원5_상부 = 기준값_원5_상부 - 측정값_원5_상부

    측정값_원6_하부_Y = np.round(np.random.uniform(1.913, 2.185, num_samples), 3)
    기준값_원6_하부_Y = np.full(num_samples, 2)
    상한공차_원6_하부_Y = np.full(num_samples, 0.1)
    하한공차_원6_하부_Y = np.full(num_samples, -0.1)
    편차_원6_하부_Y = 기준값_원6_하부_Y - 측정값_원6_하부_Y

    측정값_원6_하부_D = np.round(np.random.uniform(3.432, 3.577, num_samples), 3)
    기준값_원6_하부_D = np.full(num_samples, 3.5)
    상한공차_원6_하부_D = np.full(num_samples, 0.1)
    하한공차_원6_하부_D = np.full(num_samples, -0.1)
    편차_원6_하부_D = 기준값_원6_하부_D - 측정값_원6_하부_D

    측정값_거리3 = np.round(np.random.uniform(46.379, 46.624, num_samples), 3)
    기준값_거리3 = np.full(num_samples, 46.5)
    상한공차_거리3 = np.full(num_samples, 0.3)
    하한공차_거리3 = np.full(num_samples, -0.3)
    편차_거리3 = 기준값_거리3 - 측정값_거리3
    
    # 평면2_Z 데이터 생성
    측정값_평면2_Z = np.round(np.random.uniform(12.224, 12.601, num_samples), 3)
    기준값_평면2_Z = np.full(num_samples, 12)
    상한공차_평면2_Z = np.full(num_samples, 0.5)
    하한공차_평면2_Z = np.full(num_samples, 0)
    편차_평면2_Z = 측정값_평면2_Z - 기준값_평면2_Z
    
    # 평면2_평면도 데이터 생성
    측정값_평면2_평면도 = np.round(np.random.uniform(0.006, 1.528, num_samples), 3)
    기준값_평면2_평면도 = np.full(num_samples, 0.05)
    상한공차_평면2_평면도 = np.full(num_samples, 0.05)
    하한공차_평면2_평면도 = np.full(num_samples, 0)
    편차_평면2_평면도 = 기준값_평면2_평면도 - 측정값_평면2_평면도

    측정값_평면2_평행도 = np.round(np.random.uniform(0.018, 2.035, num_samples), 3)
    기준값_평면2_평행도 = np.full(num_samples, 0.05)
    상한공차_평면2_평행도 = np.full(num_samples, 0.05)
    하한공차_평면2_평행도 = np.full(num_samples, 0)
    편차_평면2_평행도 = 기준값_평면2_평행도 - 측정값_평면2_평행도
    
    # 평면3_Z 데이터 생성
    측정값_평면3_Z = np.round(np.random.uniform(12.187, 12.455, num_samples), 3)
    기준값_평면3_Z = np.full(num_samples, 11.8)
    상한공차_평면3_Z = np.full(num_samples, -0.05)
    하한공차_평면3_Z = np.full(num_samples, -0.1)
    편차_평면3_Z = 측정값_평면3_Z - 기준값_평면3_Z
    
    # 평면3_평면도 데이터 생성
    측정값_평면3_평면도 = np.round(np.random.uniform(0.003, 0.325, num_samples), 3)
    기준값_평면3_평면도 = np.full(num_samples, 0.05)
    상한공차_평면3_평면도 = np.full(num_samples, 0.05)
    하한공차_평면3_평면도 = np.full(num_samples, 0)
    편차_평면3_평면도 = 기준값_평면3_평면도 - 측정값_평면3_평면도

    측정값_평면3_평행도 = np.round(np.random.uniform(0.026, 0.468, num_samples), 3)
    기준값_평면3_평행도 = np.full(num_samples, 0.05)
    상한공차_평면3_평행도 = np.full(num_samples, 0.05)
    하한공차_평면3_평행도 = np.full(num_samples, 0)
    편차_평면3_평행도 = 기준값_평면3_평행도 - 측정값_평면3_평행도
    
    # 원7(E)의 되부름_X 데이터 생성
    측정값_원7_E_X = np.round(np.random.uniform(-0.079, 0.08, num_samples), 3)
    기준값_원7_E_X = np.full(num_samples, 0)
    상한공차_원7_E_X = np.full(num_samples, 0.2)
    하한공차_원7_E_X = np.full(num_samples, -0.2)
    편차_원7_E_X = 측정값_원7_E_X - 기준값_원7_E_X
    
    # 원7(E)의 되부름_Y 데이터 생성
    측정값_원7_E_Y = np.round(np.random.uniform(-0.006, 0.239, num_samples), 3)
    기준값_원7_E_Y = np.full(num_samples, 0)
    상한공차_원7_E_Y = np.full(num_samples, 0.2)
    하한공차_원7_E_Y = np.full(num_samples, -0.2)
    편차_원7_E_Y = 측정값_원7_E_Y - 기준값_원7_E_Y

    측정값_원7_E_D = np.round(np.random.uniform(25.653, 25.732, num_samples), 3)
    기준값_원7_E_D = np.full(num_samples, 26.1)
    상한공차_원7_E_D = np.full(num_samples, 0)
    하한공차_원7_E_D = np.full(num_samples, -0.5)
    편차_원7_E_D = 측정값_원7_E_D - 기준값_원7_E_D
    
    # 점28의 되부름 <소재 원점>_X 데이터 생성
    측정값_점28_X = np.round(np.random.uniform(116.603, 116.742, num_samples), 3)
    기준값_점28_X = np.full(num_samples, 116.6)
    상한공차_점28_X = np.full(num_samples, 0.1)
    하한공차_점28_X = np.full(num_samples, 0)
    편차_점28_X = 측정값_점28_X - 기준값_점28_X
    
    # 점28의 되부름 <소재 원점>_Y 데이터 생성
    측정값_점28_Y = np.round(np.random.uniform(-10.913, -10.9, num_samples), 3)
    기준값_점28_Y = np.full(num_samples, -10.9)
    상한공차_점28_Y = np.full(num_samples, 0.1)
    하한공차_점28_Y = np.full(num_samples, -0.1)
    편차_점28_Y = 측정값_점28_Y - 기준값_점28_Y

    측정값_각도2 = np.round(np.random.uniform(56.555, 57.798, num_samples), 3)
    기준값_각도2 = np.full(num_samples, 57.121)
    상한공차_각도2 = np.full(num_samples, 0.333)
    하한공차_각도2 = np.full(num_samples, -0.333)
    편차_각도2 = np.round(np.random.uniform(-0.566, 0.677, num_samples), 3)
    
    # 점29의 되부름 <소재원점>_X 데이터 생성
    측정값_점29_X = np.round(np.random.uniform(72.891, 72.962, num_samples), 3)
    기준값_점29_X = np.full(num_samples, 72.87)
    상한공차_점29_X = np.full(num_samples, 0.1)
    하한공차_점29_X = np.full(num_samples, -0.03)
    편차_점29_X = np.round(np.random.uniform(0.021, 0.092, num_samples), 3)
    
    # 점29의 되부름 <소재원점>_Y 데이터 생성
    측정값_점29_Y = np.round(np.random.uniform(-2.42, -2.341, num_samples), 3)
    기준값_점29_Y = np.full(num_samples, -2.3)
    상한공차_점29_Y = np.full(num_samples, 0.1)
    하한공차_점29_Y = np.full(num_samples, -0.1)
    편차_점29_Y = np.round(np.random.uniform(0.041, 0.12, num_samples), 3)

    측정값_점30_X = np.round(np.random.uniform(72.893, 72.99, num_samples), 3)
    기준값_점30_X = np.full(num_samples, 72.87)
    상한공차_점30_X = np.full(num_samples, 0.1)
    하한공차_점30_X = np.full(num_samples, -0.03)
    편차_점30_X = np.round(np.random.uniform(0.023, 0.12, num_samples), 3)
    
    # 점30의 되부름 <소재원점>_Y 데이터 생성
    측정값_점30_Y = np.round(np.random.uniform(-2.428, -2.265, num_samples), 3)
    기준값_점30_Y = np.full(num_samples, -2.3)
    상한공차_점30_Y = np.full(num_samples, 0.1)
    하한공차_점30_Y = np.full(num_samples, -0.1)
    편차_점30_Y = np.round(np.random.uniform(-0.035, 0.128, num_samples), 3)
    
    # 직선25의 되부름_X/Y 데이터 생성
    측정값_직선25 = np.round(np.random.uniform(-15.453, -13.336, num_samples), 3)
    기준값_직선25 = np.full(num_samples, -14.5)
    상한공차_직선25 = np.full(num_samples, 0.5)
    하한공차_직선25 = np.full(num_samples, -0.5)
    편차_직선25 = np.round(np.random.uniform(-1.164, 0.953, num_samples), 3)

    측정값_직선26 = np.round(np.random.uniform(13.635, 15.094, num_samples), 3)
    기준값_직선26 = np.full(num_samples, 14.5)
    상한공차_직선26 = np.full(num_samples, 0.5)
    하한공차_직선26 = np.full(num_samples, -0.5)
    편차_직선26 = np.round(np.random.uniform(-0.865, 0.594, num_samples), 3)
    
    # 거리4의 XAXIS[평균]:점32와 점31 <소재기준>_DS 데이터 생성
    측정값_거리4 = np.round(np.random.uniform(10.287, 10.391, num_samples), 3)
    기준값_거리4 = np.full(num_samples, 10.3)
    상한공차_거리4 = np.full(num_samples, 0.1)
    하한공차_거리4 = np.full(num_samples, -0.1)
    편차_거리4 = np.round(np.random.uniform(-0.013, 0.091, num_samples), 3)
    

    return (측정값_원1_I_D, 기준값_원1_I_D, 상한공차_원1_I_D, 하한공차_원1_I_D, 편차_원1_I_D,
            측정값_원2_I_D, 기준값_원2_I_D, 상한공차_원2_I_D, 하한공차_원2_I_D, 편차_원2_I_D,
            측정값_원3_I_D, 기준값_원3_I_D, 상한공차_원3_I_D, 하한공차_원3_I_D, 편차_원3_I_D,
            측정값_원통1_I_D, 기준값_원통1_I_D, 상한공차_원통1_I_D, 하한공차_원통1_I_D, 편차_원통1_I_D,
            측정값_원통1_I_직각도, 기준값_원통1_I_직각도, 상한공차_원통1_I_직각도, 하한공차_원통1_I_직각도, 편차_원통1_I_직각도,
            측정값_점2_X, 기준값_점2_X, 상한공차_점2_X, 하한공차_점2_X, 편차_점2_X,
            측정값_점2_Y, 기준값_점2_Y, 상한공차_점2_Y, 하한공차_점2_Y, 편차_점2_Y,
            측정값_각도1, 기준값_각도1, 상한공차_각도1, 하한공차_각도1, 편차_각도1,
            측정값_직선4, 기준값_직선4, 상한공차_직선4, 하한공차_직선4, 편차_직선4,
            측정값_직선5, 기준값_직선5, 상한공차_직선5, 하한공차_직선5, 편차_직선5,
            측정값_점3, 기준값_점3, 상한공차_점3, 하한공차_점3, 편차_점3,
            측정값_직선6, 기준값_직선6, 상한공차_직선6, 하한공차_직선6, 편차_직선6,
            측정값_점4_X, 기준값_점4_X, 상한공차_점4_X, 하한공차_점4_X, 편차_점4_X,
            측정값_점4_Y, 기준값_점4_Y, 상한공차_점4_Y, 하한공차_점4_Y, 편차_점4_Y,
            측정값_직선7, 기준값_직선7, 상한공차_직선7, 하한공차_직선7, 편차_직선7,
            측정값_직선8, 기준값_직선8, 상한공차_직선8, 하한공차_직선8, 편차_직선8,
            측정값_점5_X, 기준값_점5_X, 상한공차_점5_X, 하한공차_점5_X, 편차_점5_X,
            측정값_점5_Y, 기준값_점5_Y, 상한공차_점5_Y, 하한공차_점5_Y, 편차_점5_Y,
            측정값_원4_X, 기준값_원4_X, 상한공차_원4_X, 하한공차_원4_X, 편차_원4_X,
            측정값_원4_E_소재_Y, 기준값_원4_E_소재_Y, 상한공차_원4_E_소재_Y, 하한공차_원4_E_소재_Y, 편차_원4_E_소재_Y,
            측정값_원4_E_소재_D, 기준값_원4_E_소재_D, 상한공차_원4_E_소재_D, 하한공차_원4_E_소재_D, 편차_원4_E_소재_D,
            측정값_점6_소재_X, 기준값_점6_소재_X, 상한공차_점6_소재_X, 하한공차_점6_소재_X, 편차_점6_소재_X,
            측정값_점6_소재_Y, 기준값_점6_소재_Y, 상한공차_점6_소재_Y, 하한공차_점6_소재_Y, 편차_점6_소재_Y,
            측정값_점7_X, 기준값_점7_X, 상한공차_점7_X, 하한공차_점7_X, 편차_점7_X,
            측정값_점7_Y, 기준값_점7_Y, 상한공차_점7_Y, 하한공차_점7_Y, 편차_점7_Y,
            측정값_점8_X, 기준값_점8_X, 상한공차_점8_X, 하한공차_점8_X, 편차_점8_X,
            측정값_직선14, 기준값_직선14, 상한공차_직선14, 하한공차_직선14, 편차_직선14,
            측정값_직선16, 기준값_직선16, 상한공차_직선16, 하한공차_직선16, 편차_직선16,
            측정값_직선18, 기준값_직선18, 상한공차_직선18, 하한공차_직선18, 편차_직선18,
            측정값_거리1, 기준값_거리1, 상한공차_거리1, 하한공차_거리1, 편차_거리1,
            측정값_점13, 기준값_점13, 상한공차_점13, 하한공차_점13, 편차_점13,
            측정값_직선19, 기준값_직선19, 상한공차_직선19, 하한공차_직선19, 편차_직선19,
            측정값_직선21, 기준값_직선21, 상한공차_직선21, 하한공차_직선21, 편차_직선21,
            측정값_거리2, 기준값_거리2, 상한공차_거리2, 하한공차_거리2, 편차_거리2,
            측정값_점18, 기준값_점18, 상한공차_점18, 하한공차_점18, 편차_점18,
            측정값_점19, 기준값_점19, 상한공차_점19, 하한공차_점19, 편차_점19,
            측정값_점20, 기준값_점20, 상한공차_점20, 하한공차_점20, 편차_점20,
            측정값_원5, 기준값_원5, 상한공차_원5, 하한공차_원5, 편차_원5,
            측정값_원5_상부, 기준값_원5_상부, 상한공차_원5_상부, 하한공차_원5_상부, 편차_원5_상부,
            측정값_원6_하부_Y, 기준값_원6_하부_Y, 상한공차_원6_하부_Y, 하한공차_원6_하부_Y, 편차_원6_하부_Y,
            측정값_원6_하부_D, 기준값_원6_하부_D, 상한공차_원6_하부_D, 하한공차_원6_하부_D, 편차_원6_하부_D,
            측정값_거리3, 기준값_거리3, 상한공차_거리3, 하한공차_거리3, 편차_거리3,
            측정값_평면2_Z, 기준값_평면2_Z, 상한공차_평면2_Z, 하한공차_평면2_Z, 편차_평면2_Z,
            측정값_평면2_평면도, 기준값_평면2_평면도, 상한공차_평면2_평면도, 하한공차_평면2_평면도, 편차_평면2_평면도,
            측정값_평면2_평행도, 기준값_평면2_평행도, 상한공차_평면2_평행도, 하한공차_평면2_평행도, 편차_평면2_평행도,
            측정값_평면3_Z, 기준값_평면3_Z, 상한공차_평면3_Z, 하한공차_평면3_Z, 편차_평면3_Z,
            측정값_평면3_평면도, 기준값_평면3_평면도, 상한공차_평면3_평면도, 하한공차_평면3_평면도, 편차_평면3_평면도,
            측정값_평면3_평행도, 기준값_평면3_평행도, 상한공차_평면3_평행도, 하한공차_평면3_평행도, 편차_평면3_평행도,
            측정값_원7_E_X, 기준값_원7_E_X, 상한공차_원7_E_X, 하한공차_원7_E_X, 편차_원7_E_X,
            측정값_원7_E_Y, 기준값_원7_E_Y, 상한공차_원7_E_Y, 하한공차_원7_E_Y, 편차_원7_E_Y,
            측정값_원7_E_D, 기준값_원7_E_D, 상한공차_원7_E_D, 하한공차_원7_E_D, 편차_원7_E_D,
            측정값_점28_X, 기준값_점28_X, 상한공차_점28_X, 하한공차_점28_X, 편차_점28_X,
            측정값_점28_Y, 기준값_점28_Y, 상한공차_점28_Y, 하한공차_점28_Y, 편차_점28_Y,
            측정값_각도2, 기준값_각도2, 상한공차_각도2, 하한공차_각도2, 편차_각도2,
            측정값_점29_X, 기준값_점29_X, 상한공차_점29_X, 하한공차_점29_X, 편차_점29_X,
            측정값_점29_Y, 기준값_점29_Y, 상한공차_점29_Y, 하한공차_점29_Y, 편차_점29_Y,
            측정값_점30_X, 기준값_점30_X, 상한공차_점30_X, 하한공차_점30_X, 편차_점30_X,
            측정값_점30_Y, 기준값_점30_Y, 상한공차_점30_Y, 하한공차_점30_Y, 편차_점30_Y,
            측정값_직선25, 기준값_직선25, 상한공차_직선25, 하한공차_직선25, 편차_직선25,
            측정값_직선26, 기준값_직선26, 상한공차_직선26, 하한공차_직선26, 편차_직선26,
            측정값_거리4, 기준값_거리4, 상한공차_거리4, 하한공차_거리4, 편차_거리4
            )


# 데이터 샘플링
num_samples = 10
측정값_평면1_평면도1, 기준값_평면1_평면도, 상한공차_평면1_평면도, 하한공차_평면1_평면도, 편차_평면1_평면도 =sample_data(num_samples)
(측정값_원1_I_D, 기준값_원1_I_D, 상한공차_원1_I_D, 하한공차_원1_I_D, 편차_원1_I_D,
 측정값_원2_I_D, 기준값_원2_I_D, 상한공차_원2_I_D, 하한공차_원2_I_D, 편차_원2_I_D,
 측정값_원3_I_D, 기준값_원3_I_D, 상한공차_원3_I_D, 하한공차_원3_I_D, 편차_원3_I_D,
 측정값_원통1_I_D, 기준값_원통1_I_D, 상한공차_원통1_I_D, 하한공차_원통1_I_D, 편차_원통1_I_D,
 측정값_원통1_I_직각도, 기준값_원통1_I_직각도, 상한공차_원통1_I_직각도, 하한공차_원통1_I_직각도, 편차_원통1_I_직각도,
 측정값_점2_X, 기준값_점2_X, 상한공차_점2_X, 하한공차_점2_X, 편차_점2_X,
 측정값_점2_Y, 기준값_점2_Y, 상한공차_점2_Y, 하한공차_점2_Y, 편차_점2_Y,
 측정값_각도1, 기준값_각도1, 상한공차_각도1, 하한공차_각도1, 편차_각도1,
 측정값_직선4, 기준값_직선4, 상한공차_직선4, 하한공차_직선4, 편차_직선4,
 측정값_평면1_평면도, 기준값_평면1_평면도, 상한공차_평면1_평면도, 하한공차_평면1_평면도, 편차_평면1_평면도,
 측정값_직선5, 기준값_직선5, 상한공차_직선5, 하한공차_직선5, 편차_직선5,
 측정값_점3, 기준값_점3, 상한공차_점3, 하한공차_점3, 편차_점3,
 측정값_직선6, 기준값_직선6, 상한공차_직선6, 하한공차_직선6, 편차_직선6,
 측정값_점4_X, 기준값_점4_X, 상한공차_점4_X, 하한공차_점4_X, 편차_점4_X,
 측정값_점4_Y, 기준값_점4_Y, 상한공차_점4_Y, 하한공차_점4_Y, 편차_점4_Y,
 측정값_직선7, 기준값_직선7, 상한공차_직선7, 하한공차_직선7, 편차_직선7,
 측정값_직선8, 기준값_직선8, 상한공차_직선8, 하한공차_직선8, 편차_직선8,
 측정값_점5_X, 기준값_점5_X, 상한공차_점5_X, 하한공차_점5_X, 편차_점5_X,
 측정값_점5_Y, 기준값_점5_Y, 상한공차_점5_Y, 하한공차_점5_Y, 편차_점5_Y,
 측정값_원4_X, 기준값_원4_X, 상한공차_원4_X, 하한공차_원4_X, 편차_원4_X,
 측정값_원4_E_소재_Y, 기준값_원4_E_소재_Y, 상한공차_원4_E_소재_Y, 하한공차_원4_E_소재_Y, 편차_원4_E_소재_Y,
 측정값_원4_E_소재_D, 기준값_원4_E_소재_D, 상한공차_원4_E_소재_D, 하한공차_원4_E_소재_D, 편차_원4_E_소재_D,
 측정값_점6_소재_X, 기준값_점6_소재_X, 상한공차_점6_소재_X, 하한공차_점6_소재_X, 편차_점6_소재_X,
 측정값_점6_소재_Y, 기준값_점6_소재_Y, 상한공차_점6_소재_Y, 하한공차_점6_소재_Y, 편차_점6_소재_Y,
 측정값_점7_X, 기준값_점7_X, 상한공차_점7_X, 하한공차_점7_X, 편차_점7_X,
 측정값_점7_Y, 기준값_점7_Y, 상한공차_점7_Y, 하한공차_점7_Y, 편차_점7_Y,
 측정값_점8_X, 기준값_점8_X, 상한공차_점8_X, 하한공차_점8_X, 편차_점8_X,
 측정값_직선14, 기준값_직선14, 상한공차_직선14, 하한공차_직선14, 편차_직선14,
 측정값_직선16, 기준값_직선16, 상한공차_직선16, 하한공차_직선16, 편차_직선16,
 측정값_직선18, 기준값_직선18, 상한공차_직선18, 하한공차_직선18, 편차_직선18,
 측정값_거리1, 기준값_거리1, 상한공차_거리1, 하한공차_거리1, 편차_거리1,
 측정값_점13, 기준값_점13, 상한공차_점13, 하한공차_점13, 편차_점13,
 측정값_직선19, 기준값_직선19, 상한공차_직선19, 하한공차_직선19, 편차_직선19,
 측정값_직선21, 기준값_직선21, 상한공차_직선21, 하한공차_직선21, 편차_직선21,
 측정값_거리2, 기준값_거리2, 상한공차_거리2, 하한공차_거리2, 편차_거리2,
 측정값_점18, 기준값_점18, 상한공차_점18, 하한공차_점18, 편차_점18,
 측정값_점19, 기준값_점19, 상한공차_점19, 하한공차_점19, 편차_점19,
 측정값_점20, 기준값_점20, 상한공차_점20, 하한공차_점20, 편차_점20,
 측정값_원5, 기준값_원5, 상한공차_원5, 하한공차_원5, 편차_원5,
 측정값_원5_상부, 기준값_원5_상부, 상한공차_원5_상부, 하한공차_원5_상부, 편차_원5_상부,
 측정값_원6_하부_Y, 기준값_원6_하부_Y, 상한공차_원6_하부_Y, 하한공차_원6_하부_Y, 편차_원6_하부_Y,
 측정값_원6_하부_D, 기준값_원6_하부_D, 상한공차_원6_하부_D, 하한공차_원6_하부_D, 편차_원6_하부_D,
 측정값_거리3, 기준값_거리3, 상한공차_거리3, 하한공차_거리3, 편차_거리3,
 측정값_평면2_Z, 기준값_평면2_Z, 상한공차_평면2_Z, 하한공차_평면2_Z, 편차_평면2_Z,
 측정값_평면2_평면도, 기준값_평면2_평면도, 상한공차_평면2_평면도, 하한공차_평면2_평면도, 편차_평면2_평면도,
 측정값_평면2_평행도, 기준값_평면2_평행도, 상한공차_평면2_평행도, 하한공차_평면2_평행도, 편차_평면2_평행도,
 측정값_평면3_Z, 기준값_평면3_Z, 상한공차_평면3_Z, 하한공차_평면3_Z, 편차_평면3_Z,
 측정값_평면3_평면도, 기준값_평면3_평면도, 상한공차_평면3_평면도, 하한공차_평면3_평면도, 편차_평면3_평면도,
 측정값_평면3_평행도, 기준값_평면3_평행도, 상한공차_평면3_평행도, 하한공차_평면3_평행도, 편차_평면3_평행도,
 측정값_원7_E_X, 기준값_원7_E_X, 상한공차_원7_E_X, 하한공차_원7_E_X, 편차_원7_E_X,
 측정값_원7_E_Y, 기준값_원7_E_Y, 상한공차_원7_E_Y, 하한공차_원7_E_Y, 편차_원7_E_Y,
 측정값_원7_E_D, 기준값_원7_E_D, 상한공차_원7_E_D, 하한공차_원7_E_D, 편차_원7_E_D,
 측정값_점28_X, 기준값_점28_X, 상한공차_점28_X, 하한공차_점28_X, 편차_점28_X,
 측정값_점28_Y, 기준값_점28_Y, 상한공차_점28_Y, 하한공차_점28_Y, 편차_점28_Y,
 측정값_각도2, 기준값_각도2, 상한공차_각도2, 하한공차_각도2, 편차_각도2,
 측정값_점29_X, 기준값_점29_X, 상한공차_점29_X, 하한공차_점29_X, 편차_점29_X,
 측정값_점29_Y, 기준값_점29_Y, 상한공차_점29_Y, 하한공차_점29_Y, 편차_점29_Y,
 측정값_점30_X, 기준값_점30_X, 상한공차_점30_X, 하한공차_점30_X, 편차_점30_X,
 측정값_점30_Y, 기준값_점30_Y, 상한공차_점30_Y, 하한공차_점30_Y, 편차_점30_Y,
 측정값_직선25, 기준값_직선25, 상한공차_직선25, 하한공차_직선25, 편차_직선25,
 측정값_직선26, 기준값_직선26, 상한공차_직선26, 하한공차_직선26, 편차_직선26,
 측정값_거리4, 기준값_거리4, 상한공차_거리4, 하한공차_거리4, 편차_거리4
 ) = additional_sample_data(num_samples)

# 데이터 프레임 생성
df = pd.DataFrame({
    '측정값_평면1_평면도': 측정값_평면1_평면도1,
    '기준값_평면1_평면도': 기준값_평면1_평면도,
    '상한공차_평면1_평면도': 상한공차_평면1_평면도,
    '하한공차_평면1_평면도': 하한공차_평면1_평면도,
    '편차_평면1_평면도': 편차_평면1_평면도,
    '측정값_원1(I)_D': 측정값_원1_I_D,
    '기준값_원1(I)_D': 기준값_원1_I_D,
    '상한공차_원1(I)_D': 상한공차_원1_I_D,
    '하한공차_원1(I)_D': 하한공차_원1_I_D,
    '편차_원1(I)_D': 편차_원1_I_D,
    '측정값_원2(I)_D': 측정값_원2_I_D,
    '기준값_원2(I)_D': 기준값_원2_I_D,
    '상한공차_원2(I)_D': 상한공차_원2_I_D,
    '하한공차_원2(I)_D': 하한공차_원2_I_D,
    '편차_원2(I)_D': 편차_원2_I_D,
    '측정값_원3(I)_D': 측정값_원3_I_D,
    '기준값_원3(I)_D': 기준값_원3_I_D,
    '상한공차_원3(I)_D': 상한공차_원3_I_D,
    '하한공차_원3(I)_D': 하한공차_원3_I_D,
    '편차_원3(I)_D': 편차_원3_I_D,
    '측정값_원통1(I)_D': 측정값_원통1_I_D,
    '기준값_원통1(I)_D': 기준값_원통1_I_D,
    '상한공차_원통1(I)_D': 상한공차_원통1_I_D,
    '하한공차_원통1(I)_D': 하한공차_원통1_I_D,
    '편차_원통1(I)_D': 편차_원통1_I_D,
    '측정값_원통1(I)_직각도': 측정값_원통1_I_직각도,
    '기준값_원통1(I)_직각도': 기준값_원통1_I_직각도,
    '상한공차_원통1(I)_직각도': 상한공차_원통1_I_직각도,
    '하한공차_원통1(I)_직각도': 하한공차_원통1_I_직각도,
    '편차_원통1(I)_직각도': 편차_원통1_I_직각도,
    '측정값_점2_X': 측정값_점2_X,
    '기준값_점2_X': 기준값_점2_X,
    '상한공차_점2_X': 상한공차_점2_X,
    '하한공차_점2_X': 하한공차_점2_X,
    '편차_점2_X': 편차_점2_X,
    '측정값_점2_Y': 측정값_점2_Y,
    '기준값_점2_Y': 기준값_점2_Y,
    '상한공차_점2_Y': 상한공차_점2_Y,
    '하한공차_점2_Y': 하한공차_점2_Y,
    '편차_점2_Y': 편차_점2_Y,
    '측정값_각도1': 측정값_각도1,
    '기준값_각도1': 기준값_각도1,
    '상한공차_각도1': 상한공차_각도1,
    '하한공차_각도1': 하한공차_각도1,
    '편차_각도1': 편차_각도1,
    '측정값_직선4': 측정값_직선4,
    '기준값_직선4': 기준값_직선4,
    '상한공차_직선4': 상한공차_직선4,
    '하한공차_직선4': 하한공차_직선4,
    '편차_직선4': 편차_직선4,
    '측정값_직선5': 측정값_직선5,
    '기준값_직선5': 기준값_직선5,
    '상한공차_직선5': 상한공차_직선5,
    '하한공차_직선5': 하한공차_직선5,
    '편차_직선5': 편차_직선5,
    '측정값_점3': 측정값_점3,
    '기준값_점3': 기준값_점3,
    '상한공차_점3': 상한공차_점3,
    '하한공차_점3': 하한공차_점3,
    '편차_점3': 편차_점3,
    '측정값_직선6': 측정값_직선6,
    '기준값_직선6': 기준값_직선6,
    '상한공차_직선6': 상한공차_직선6,
    '하한공차_직선6': 하한공차_직선6,
    '편차_직선6': 편차_직선6,
    '측정값_점4_X': 측정값_점4_X,
    '기준값_점4_X': 기준값_점4_X,
    '상한공차_점4_X': 상한공차_점4_X,
    '하한공차_점4_X': 하한공차_점4_X,
    '편차_점4_X': 편차_점4_X,
    '측정값_점4_Y': 측정값_점4_Y,
    '기준값_점4_Y': 기준값_점4_Y,
    '상한공차_점4_Y': 상한공차_점4_Y,
    '하한공차_점4_Y': 하한공차_점4_Y,
    '편차_점4_Y': 편차_점4_Y,
    '측정값_직선7': 측정값_직선7,
    '기준값_직선7': 기준값_직선7,
    '상한공차_직선7': 상한공차_직선7,
    '하한공차_직선7': 하한공차_직선7,
    '편차_직선7': 편차_직선7,
    '측정값_직선8': 측정값_직선8,
    '기준값_직선8': 기준값_직선8,
    '상한공차_직선8': 상한공차_직선8,
    '하한공차_직선8': 하한공차_직선8,
    '편차_직선8': 편차_직선8,
    '측정값_점5_X': 측정값_점5_X,
    '기준값_점5_X': 기준값_점5_X,
    '상한공차_점5_X': 상한공차_점5_X,
    '하한공차_점5_X': 하한공차_점5_X,
    '편차_점5_X': 편차_점5_X,
    '측정값_점5_Y': 측정값_점5_Y,
    '기준값_점5_Y': 기준값_점5_Y,
    '상한공차_점5_Y': 상한공차_점5_Y,
    '하한공차_점5_Y': 하한공차_점5_Y,
    '편차_점5_Y': 편차_점5_Y,
    '측정값_원4_X': 측정값_원4_X,
    '기준값_원4_X': 기준값_원4_X,
    '상한공차_원4_X': 상한공차_원4_X,
    '하한공차_원4_X': 하한공차_원4_X,
    '편차_원4_X': 편차_원4_X,
    '측정값_원4(E) <소재>_Y': 측정값_원4_E_소재_Y,
    '기준값_원4(E) <소재>_Y': 기준값_원4_E_소재_Y,
    '상한공차_원4(E) <소재>_Y': 상한공차_원4_E_소재_Y,
    '하한공차_원4(E) <소재>_Y': 하한공차_원4_E_소재_Y,
    '편차_원4(E) <소재>_Y': 편차_원4_E_소재_Y,
    '측정값_원4(E) <소재>_D': 측정값_원4_E_소재_D,
    '기준값_원4(E) <소재>_D': 기준값_원4_E_소재_D,
    '상한공차_원4(E) <소재>_D': 상한공차_원4_E_소재_D,
    '하한공차_원4(E) <소재>_D': 하한공차_원4_E_소재_D,
    '편차_원4(E) <소재>_D': 편차_원4_E_소재_D,
    '측정값_점6 <- 직선9와 직선10의 교차점 <소재>_X': 측정값_점6_소재_X,
    '기준값_점6 <- 직선9와 직선10의 교차점 <소재>_X': 기준값_점6_소재_X,
    '상한공차_점6 <- 직선9와 직선10의 교차점 <소재>_X': 상한공차_점6_소재_X,
    '하한공차_점6 <- 직선9와 직선10의 교차점 <소재>_X': 하한공차_점6_소재_X,
    '편차_점6 <- 직선9와 직선10의 교차점 <소재>_X': 편차_점6_소재_X,
    '측정값_점6 <- 직선9와 직선10의 교차점 <소재>_Y': 측정값_점6_소재_Y,
    '기준값_점6 <- 직선9와 직선10의 교차점 <소재>_Y': 기준값_점6_소재_Y,
    '상한공차_점6 <- 직선9와 직선10의 교차점 <소재>_Y': 상한공차_점6_소재_Y,
    '하한공차_점6 <- 직선9와 직선10의 교차점 <소재>_Y': 하한공차_점6_소재_Y,
    '편차_점6 <- 직선9와 직선10의 교차점 <소재>_Y': 편차_점6_소재_Y,
    '측정값_점7_X': 측정값_점7_X,
    '기준값_점7_X': 기준값_점7_X,
    '상한공차_점7_X': 상한공차_점7_X,
    '하한공차_점7_X': 하한공차_점7_X,
    '편차_점7_X': 편차_점7_X,
    '측정값_점7_Y': 측정값_점7_Y,
    '기준값_점7_Y': 기준값_점7_Y,
    '상한공차_점7_Y': 상한공차_점7_Y,
    '하한공차_점7_Y': 하한공차_점7_Y,
    '편차_점7_Y': 편차_점7_Y,
    '측정값_점8_X': 측정값_점8_X,
    '기준값_점8_X': 기준값_점8_X,
    '상한공차_점8_X': 상한공차_점8_X,
    '하한공차_점8_X': 하한공차_점8_X,
    '편차_점8_X': 편차_점8_X,
    '측정값_직선14 <23.1° 소재>_Y/X': 측정값_직선14,
    '기준값_직선14 <23.1° 소재>_Y/X': 기준값_직선14,
    '상한공차_직선14 <23.1° 소재>_Y/X': 상한공차_직선14,
    '하한공차_직선14 <23.1° 소재>_Y/X': 하한공차_직선14,
    '편차_직선14 <23.1° 소재>_Y/X': 편차_직선14,
    '측정값_직선16 <우상 소재>_X/Y': 측정값_직선16,
    '기준값_직선16 <우상 소재>_X/Y': 기준값_직선16,
    '상한공차_직선16 <우상 소재>_X/Y': 상한공차_직선16,
    '하한공차_직선16 <우상 소재>_X/Y': 하한공차_직선16,
    '편차_직선16 <우상 소재>_X/Y': 편차_직선16,
    '측정값_직선18 <좌상 소재>_X/Y': 측정값_직선18,
    '기준값_직선18 <좌상 소재>_X/Y': 기준값_직선18,
    '상한공차_직선18 <좌상 소재>_X/Y': 상한공차_직선18,
    '하한공차_직선18 <좌상 소재>_X/Y': 하한공차_직선18,
    '편차_직선18 <좌상 소재>_X/Y': 편차_직선18,
    '측정값_거리1': 측정값_거리1,
    '기준값_거리1': 기준값_거리1,
    '상한공차_거리1': 상한공차_거리1,
    '하한공차_거리1': 하한공차_거리1,
    '편차_거리1': 편차_거리1,
    '측정값_점13': 측정값_점13,
    '기준값_점13': 기준값_점13,
    '상한공차_점13': 상한공차_점13,
    '하한공차_점13': 하한공차_점13,
    '편차_점13': 편차_점13,
    '측정값_직선19': 측정값_직선19,
    '기준값_직선19': 기준값_직선19,
    '상한공차_직선19': 상한공차_직선19,
    '하한공차_직선19': 하한공차_직선19,
    '편차_직선19': 편차_직선19,
    '측정값_직선21': 측정값_직선21,
    '기준값_직선21': 기준값_직선21,
    '상한공차_직선21': 상한공차_직선21,
    '하한공차_직선21': 하한공차_직선21,
    '편차_직선21': 편차_직선21,
    '측정값_거리2': 측정값_거리2,
    '기준값_거리2': 기준값_거리2,
    '상한공차_거리2': 상한공차_거리2,
    '하한공차_거리2': 하한공차_거리2,
    '편차_거리2': 편차_거리2,
    '측정값_점18': 측정값_점18,
    '기준값_점18': 기준값_점18,
    '상한공차_점18': 상한공차_점18,
    '하한공차_점18': 하한공차_점18,
    '편차_점18': 편차_점18,
    '측정값_점19': 측정값_점19,
    '기준값_점19': 기준값_점19,
    '상한공차_점19': 상한공차_점19,
    '하한공차_점19': 하한공차_점19,
    '편차_점19': 편차_점19,
    '측정값_점20': 측정값_점20,
    '기준값_점20': 기준값_점20,
    '상한공차_점20': 상한공차_점20,
    '하한공차_점20': 하한공차_점20,
    '편차_점20': 편차_점20,
    '측정값_원5': 측정값_원5,
    '기준값_원5': 기준값_원5,
    '상한공차_원5': 상한공차_원5,
    '하한공차_원5': 하한공차_원5,
    '편차_원5': 편차_원5,
    '측정값_원5(I)_상부': 측정값_원5_상부,
    '기준값_원5(I)_상부': 기준값_원5_상부,
    '상한공차_원5(I)_상부': 상한공차_원5_상부,
    '하한공차_원5(I)_상부': 하한공차_원5_상부,
    '편차_원5(I)_상부': 편차_원5_상부,
    '측정값_원6(I)_하부_Y': 측정값_원6_하부_Y,
    '기준값_원6(I)_하부_Y': 기준값_원6_하부_Y,
    '상한공차_원6(I)_하부_Y': 상한공차_원6_하부_Y,
    '하한공차_원6(I)_하부_Y': 하한공차_원6_하부_Y,
    '편차_원6(I)_하부_Y': 편차_원6_하부_Y,
    '측정값_원6(I)_하부_D': 측정값_원6_하부_D,
    '기준값_원6(I)_하부_D': 기준값_원6_하부_D,
    '상한공차_원6(I)_하부_D': 상한공차_원6_하부_D,
    '하한공차_원6(I)_하부_D': 하한공차_원6_하부_D,
    '편차_원6(I)_하부_D': 편차_원6_하부_D,
    '측정값_거리3': 측정값_거리3,
    '기준값_거리3': 기준값_거리3,
    '상한공차_거리3': 상한공차_거리3,
    '하한공차_거리3': 하한공차_거리3,
    '편차_거리3': 편차_거리3,
    '측정값_평면2_Z': 측정값_평면2_Z,
    '기준값_평면2_Z': 기준값_평면2_Z,
    '상한공차_평면2_Z': 상한공차_평면2_Z,
    '하한공차_평면2_Z': 하한공차_평면2_Z,
    '편차_평면2_Z': 편차_평면2_Z,
    '측정값_평면2_평면도': 측정값_평면2_평면도,
    '기준값_평면2_평면도': 기준값_평면2_평면도,
    '상한공차_평면2_평면도': 상한공차_평면2_평면도,
    '하한공차_평면2_평면도': 하한공차_평면2_평면도,
    '편차_평면2_평면도': 편차_평면2_평면도,
    '측정값_평면2_평행도': 측정값_평면2_평행도,
    '기준값_평면2_평행도': 기준값_평면2_평행도,
    '상한공차_평면2_평행도': 상한공차_평면2_평행도,
    '하한공차_평면2_평행도': 하한공차_평면2_평행도,
    '편차_평면2_평행도': 편차_평면2_평행도,
    '측정값_평면3_Z': 측정값_평면3_Z,
    '기준값_평면3_Z': 기준값_평면3_Z,
    '상한공차_평면3_Z': 상한공차_평면3_Z,
    '하한공차_평면3_Z': 하한공차_평면3_Z,
    '편차_평면3_Z': 편차_평면3_Z,
    '측정값_평면3_평면도': 측정값_평면3_평면도,
    '기준값_평면3_평면도': 기준값_평면3_평면도,
    '상한공차_평면3_평면도': 상한공차_평면3_평면도,
    '하한공차_평면3_평면도': 하한공차_평면3_평면도,
    '편차_평면3_평면도': 편차_평면3_평면도,
    '측정값_평면3_평행도': 측정값_평면3_평행도,
    '기준값_평면3_평행도': 기준값_평면3_평행도,
    '상한공차_평면3_평행도': 상한공차_평면3_평행도,
    '하한공차_평면3_평행도': 하한공차_평면3_평행도,
    '편차_평면3_평행도': 편차_평면3_평행도,
    '측정값_원7_E_X': 측정값_원7_E_X,
    '기준값_원7_E_X': 기준값_원7_E_X,
    '상한공차_원7_E_X': 상한공차_원7_E_X,
    '하한공차_원7_E_X': 하한공차_원7_E_X,
    '편차_원7_E_X': 편차_원7_E_X,
    '측정값_원7_E_Y': 측정값_원7_E_Y,
    '기준값_원7_E_Y': 기준값_원7_E_Y,
    '상한공차_원7_E_Y': 상한공차_원7_E_Y,
    '하한공차_원7_E_Y': 하한공차_원7_E_Y,
    '편차_원7_E_Y': 편차_원7_E_Y,
    '측정값_원7_E_D': 측정값_원7_E_D,
    '기준값_원7_E_D': 기준값_원7_E_D,
    '상한공차_원7_E_D': 상한공차_원7_E_D,
    '하한공차_원7_E_D': 하한공차_원7_E_D,
    '편차_원7_E_D': 편차_원7_E_D,
    '측정값_점28_X': 측정값_점28_X,
    '기준값_점28_X': 기준값_점28_X,
    '상한공차_점28_X': 상한공차_점28_X,
    '하한공차_점28_X': 하한공차_점28_X,
    '편차_점28_X': 편차_점28_X,
    '측정값_점28_Y': 측정값_점28_Y,
    '기준값_점28_Y': 기준값_점28_Y,
    '상한공차_점28_Y': 상한공차_점28_Y,
    '하한공차_점28_Y': 하한공차_점28_Y,
    '편차_점28_Y': 편차_점28_Y,
    '측정값_각도2': 측정값_각도2,
    '기준값_각도2': 기준값_각도2,
    '상한공차_각도2': 상한공차_각도2,
    '하한공차_각도2': 하한공차_각도2,
    '편차_각도2': 편차_각도2,
    '측정값_점29_X': 측정값_점29_X,
    '기준값_점29_X': 기준값_점29_X,
    '상한공차_점29_X': 상한공차_점29_X,
    '하한공차_점29_X': 하한공차_점29_X,
    '편차_점29_X': 편차_점29_X,
    '측정값_점29_Y': 측정값_점29_Y,
    '기준값_점29_Y': 기준값_점29_Y,
    '상한공차_점29_Y': 상한공차_점29_Y,
    '하한공차_점29_Y': 하한공차_점29_Y,
    '편차_점29_Y': 편차_점29_Y,
    '측정값_점30_X': 측정값_점30_X,
    '기준값_점30_X': 기준값_점30_X,
    '상한공차_점30_X': 상한공차_점30_X,
    '하한공차_점30_X': 하한공차_점30_X,
    '편차_점30_X': 편차_점30_X,
    '측정값_점30_Y': 측정값_점30_Y,
    '기준값_점30_Y': 기준값_점30_Y,
    '상한공차_점30_Y': 상한공차_점30_Y,
    '하한공차_점30_Y': 하한공차_점30_Y,
    '편차_점30_Y': 편차_점30_Y,
    '측정값_직선25': 측정값_직선25,
    '기준값_직선25': 기준값_직선25,
    '상한공차_직선25': 상한공차_직선25,
    '하한공차_직선25': 하한공차_직선25,
    '편차_직선25': 편차_직선25,
    '측정값_직선26': 측정값_직선26,
    '기준값_직선26': 기준값_직선26,
    '상한공차_직선26': 상한공차_직선26,
    '하한공차_직선26': 하한공차_직선26,
    '편차_직선26': 편차_직선26,
    '측정값_거리4': 측정값_거리4,
    '기준값_거리4': 기준값_거리4,
    '상한공차_거리4': 상한공차_거리4,
    '하한공차_거리4': 하한공차_거리4,
    '편차_거리4': 편차_거리4
})

# 데이터 CSV 파일로 저장
file_path = "C:\\git_folder\\CMM_DeepLearning_Module\\MLP\\test\\sample_data5.csv"
df.to_csv(file_path, index=False, encoding='utf-8-sig')