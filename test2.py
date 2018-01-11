# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

desired_width = 320
pd.set_option('display.width', desired_width)

# 人口算出。死亡者数が設定した範囲に入るまでランダムで避難者を振り分ける。
def PopulationCaluculation():
    while True:
        # P4 = np.random.randint(0.5*(Cap4+1), Cap4 + 1)
        # P5 = np.random.randint(0.5*(Cap5+1), Cap5 + 1)
        # P6 = np.random.randint(0.5*(Cap6+1), Cap6 + 1)
        # P7 = np.random.randint(0.5*(Cap7+1), Cap7 + 1)
        # P8 = np.random.randint(0.5*(Cap8+1), Cap8 + 1)
        # P9 = np.random.randint(0.5*(Cap9+1), Cap9 + 1)
        P4 = np.random.randint(Cap4 + 1)
        P5 = np.random.randint(Cap5 + 1)
        P6 = np.random.randint(Cap6 + 1)
        P7 = np.random.randint(Cap7 + 1)
        P8 = np.random.randint(Cap8 + 1)
        P9 = np.random.randint(Cap9 + 1)
        Dead_Population = Toatl_Population - (P4 + P5 + P6 + P7 + P8 + P9)

        if (Dead_Population<0 or Dead_Population > Toatl_Population * Upper_mortality_rate or Dead_Population < Toatl_Population * Lower_mortality_rate):
            continue
        return P4, P5, P6, P7, P8, P9, Dead_Population

# 需要計算
def Demandculation (P4, P5, P6, P7, P8, P9, current_demand):
    Demand = np.array([dx, dy, dz])
    Population = np.array([[P4], [P5], [P6], [P7], [P8], [P9]])
    Delta_Demand = Demand*Population
    Delta_Demand = np.round(Delta_Demand.flatten()).astype(np.int)
    Next_demand = current_demand + Delta_Demand
    return Next_demand

# 供給総量の計算
def Total_Supply_calculation():
    Total_Supply_x = int(round(dx * Toatl_Population * t_end))
    Total_Supply_y = int(round(dy * Toatl_Population))
    Total_Supply_z = int(round(dz * Toatl_Population * t_end))
    return Total_Supply_x,Total_Supply_y,Total_Supply_z

# トラックの積荷の乗せ方の設定
def TruckLoading(Ix,Iy,Iz,Truck_Max_Loading,Loading_Point,t):
    if (Loading_Point==0):
        while True:
            if(Ix+Iy+Iz<=Truck_Max_Loading):
                x = Ix
                y = Iy
                z = Iz
            elif(Ix==0 and Iy==0):
                x = 0
                y = 0
                if (Iz<=Truck_Max_Loading):
                    z = Iz
                else:
                    z = Truck_Max_Loading
            elif(Ix==0 and Iz==0):
                x = 0
                z = 0
                if (Iy<=Truck_Max_Loading):
                    y = Iy
                else:
                    y = Truck_Max_Loading
            elif(Iy==0 and Iz==0):
                y = 0
                z = 0
                if (Ix<=Truck_Max_Loading):
                    x = Ix
                else:
                    x = Truck_Max_Loading
            elif (Ix == 0):
                x = 0
                if(Iy + Iz <= Truck_Max_Loading):
                    y = Iy
                    z = Iz
                else:
                    y = Truck_Max_Loading * np.random.uniform(0, 1)
                    z = Truck_Max_Loading -y
            elif (Iy == 0):
                y = 0
                if(Ix + Iz <= Truck_Max_Loading):
                    x = Ix
                    z = Iz
                elif(Ix <= Truck_Max_Loading):
                    x = Ix
                    z = Truck_Max_Loading -x
                else:
                    x = Truck_Max_Loading * np.random.uniform(0.9, 1)
                    z = Truck_Max_Loading - x
            elif (Iz == 0):
                z = 0
                if (Ix + Iy <= Truck_Max_Loading):
                    x = Ix
                    y = Iy
                elif(Ix <= Truck_Max_Loading):
                    x = Ix
                    y = Truck_Max_Loading -x
                else:
                    x = Truck_Max_Loading * np.random.uniform(0.9, 1)
                    y = Truck_Max_Loading - x
            elif(Ix > Truck_Max_Loading):
                x = Truck_Max_Loading * np.random.uniform(0.9, 1)
                y = Truck_Max_Loading * np.random.uniform(0, 0.1)
                z = Truck_Max_Loading -x -y
            elif(Ix > Truck_Max_Loading * 0.5 and Ix <= Truck_Max_Loading):
                x = Truck_Max_Loading * np.random.uniform(0.5, 1)
                y = Truck_Max_Loading * np.random.uniform(0, 0.5)
                z = Truck_Max_Loading - x -y
            else:
                x = Ix
                y = Truck_Max_Loading * np.random.uniform(0, 1)
                z = Truck_Max_Loading -x -y
            x = int(round(x))
            y = int(round(y))
            z = int(round(z))
            if (Truck_Max_Loading<x+y+z or Ix<x or Iy<y or Iz<z or z<0):
                continue
            return x, y, z


    # 二箇所に行き先が分岐する場合
    elif (Loading_Point==1):
        while True:
            if(Ix==0 and Iy==0):
                x = 0
                y = 0
                if(Iz<=Truck_Max_Loading):
                    z = 0.5 * Iz
                elif(Iz>Truck_Max_Loading and Iz<Truck_Max_Loading*2):
                    z = 0.75 * Truck_Max_Loading
                else:
                    z = Truck_Max_Loading
            elif(Ix==0 and Iz==0):
                x = 0
                z = 0
                if(Iy<=Truck_Max_Loading):
                    y = 0.5 * Iy
                elif(Iy>Truck_Max_Loading and Iy<Truck_Max_Loading*2):
                    y = 0.75 * Truck_Max_Loading
                else:
                    y = Truck_Max_Loading
            elif(Iy==0 and Iz==0):
                y = 0
                z = 0
                if(Ix<=Truck_Max_Loading):
                    x = 0.5 * Ix
                elif(Ix>Truck_Max_Loading and Ix<Truck_Max_Loading*2):
                    x = 0.75 * Truck_Max_Loading
                else:
                    x = Truck_Max_Loading
            elif(Ix==0):
                x = 0
                if(Iy+Iz<=Truck_Max_Loading):
                    y = 0.5 * Iy
                    z = 0.5 * Iz
                elif(Iy+Iz>Truck_Max_Loading and Iy+Iz<=Truck_Max_Loading*2):
                    y = 0.75 * Truck_Max_Loading * np.random.uniform(0,1)
                    z = 0.75 * Truck_Max_Loading -y
                else:
                    y = Truck_Max_Loading * np.random.uniform(0,1)
                    z = Truck_Max_Loading -y
            elif(Iy==0):
                y = 0
                if(Ix+Iz<=Truck_Max_Loading):
                    x = 0.5 * Ix
                    z = 0.5 * Iz
                elif(Ix+Iz>Truck_Max_Loading and Ix+Iz<=Truck_Max_Loading*2):
                    if(Ix > Truck_Max_Loading):
                        x = 0.75 * Truck_Max_Loading * np.random.uniform(0.9, 1)
                    else:
                        x = 0.5 * Ix
                    z = 0.75 * Truck_Max_Loading -x
                else:
                    if(Ix > Truck_Max_Loading):
                        x = Truck_Max_Loading * np.random.uniform(0.9, 1)
                    else:
                        x = 0.5 * Ix
                    z = Truck_Max_Loading -x
            elif(Iz==0):
                z = 0
                if(Ix+Iy<=Truck_Max_Loading):
                    x = 0.5 * Ix
                    y = 0.5 * Iy
                elif(Ix+Iy>Truck_Max_Loading and Ix+Iy<=Truck_Max_Loading*2):
                    if(Ix > Truck_Max_Loading):
                        x = 0.75 * Truck_Max_Loading * np.random.uniform(0.9, 1)
                    else:
                        x = 0.5 * Ix
                    y = 0.75 * Truck_Max_Loading -x
                else:
                    if(Ix > Truck_Max_Loading):
                        x = Truck_Max_Loading * np.random.uniform(0.9, 1)
                    else:
                        x = 0.5 * Ix
                    y = Truck_Max_Loading -x
            elif (Ix + Iy + Iz <= Truck_Max_Loading):
                    x = 0.5 * Ix
                    y = 0.5 * Iy
                    z = 0.5 * Iz
            elif (Ix + Iy + Iz > Truck_Max_Loading and Ix + Iy + Iz < Truck_Max_Loading * 2):
                if (Ix > Truck_Max_Loading):
                    x = 0.75 * Truck_Max_Loading * np.random.uniform(0.9, 1)
                    y = 0.75 * Truck_Max_Loading * np.random.uniform(0, 0.1)
                else:
                    x = 0.5 * Ix
                    y = 0.75 * Truck_Max_Loading * np.random.uniform(0, 1)
                z = 0.75 * Truck_Max_Loading - x - y
            elif (Ix + Iy + Iz >= Truck_Max_Loading * 2):
                if (Ix > Truck_Max_Loading):
                    x = Truck_Max_Loading * np.random.uniform(0.9, 1)
                    y = Truck_Max_Loading * np.random.uniform(0, 0.1)
                else:
                    x = 0.5 * Ix
                    y = Truck_Max_Loading * np.random.uniform(0, 1)
                z = Truck_Max_Loading - x - y
            x = int(round(x))
            y = int(round(y))
            z = int(round(z))
            if (Truck_Max_Loading<x+y+z or Ix<x or Iy<y or Iz<z or z<0):
                continue
            return x, y, z


    # 三箇所に行き先が分岐する場合
    elif(Loading_Point==2 or Loading_Point==3):
        while True:
            if(Ix==0 and Iy==0):
                x = 0
                y = 0
                if(Iz<=Truck_Max_Loading):
                    z = 0.34 * Iz
                elif(Iz>Truck_Max_Loading and Iz<Truck_Max_Loading*3):
                    z = 0.66 * Truck_Max_Loading
                else:
                    z = Truck_Max_Loading
            elif(Ix==0 and Iz==0):
                x = 0
                z = 0
                if(Iy<=Truck_Max_Loading):
                    y = 0.34 * Iy
                elif(Iy>Truck_Max_Loading and Iy<Truck_Max_Loading*3):
                    y = 0.66 * Truck_Max_Loading
                else:
                    y = Truck_Max_Loading
            elif(Iy==0 and Iz==0):
                y = 0
                z = 0
                if(Ix<=Truck_Max_Loading):
                    x = 0.34 * Ix
                elif(Ix>Truck_Max_Loading and Ix<Truck_Max_Loading*3):
                    x = 0.66 * Truck_Max_Loading
                else:
                    x = Truck_Max_Loading
            elif(Ix==0):
                x = 0
                if(Iy+Iz<=Truck_Max_Loading):
                    y = 0.34 * Iy
                    z = 0.34 * Iz
                elif(Iy+Iz>Truck_Max_Loading and Iy+Iz<=Truck_Max_Loading*3):
                    y = 0.66 * Truck_Max_Loading * np.random.uniform(0, 1)
                    z = 0.66 * Truck_Max_Loading -y
                else:
                    y = Truck_Max_Loading * np.random.uniform(0,1)
                    z = Truck_Max_Loading -y
            elif(Iy==0):
                y = 0
                if(Ix+Iz<=Truck_Max_Loading):
                    x = 0.34 * Ix
                    z = 0.34 * Iz
                elif(Ix+Iz>Truck_Max_Loading and Ix+Iz<=Truck_Max_Loading*3):
                    if(Ix > Truck_Max_Loading):
                        x = 0.66 * Truck_Max_Loading * np.random.uniform(0.9, 1)
                    else:
                        x = 0.34 * Ix
                    z = 0.66 * Truck_Max_Loading -x
                else:
                    if(Ix > Truck_Max_Loading):
                        x = Truck_Max_Loading * np.random.uniform(0.9, 1)
                    else:
                        x = 0.34 * Ix
                    z = Truck_Max_Loading - x
            elif(Iz==0):
                z = 0
                if(Ix+Iy<=Truck_Max_Loading):
                    x = 0.34 * Ix
                    y = 0.34 * Iy
                elif(Ix+Iy>Truck_Max_Loading and Ix+Iy<=Truck_Max_Loading*3):
                    if(Ix > Truck_Max_Loading):
                        x = 0.66 * Truck_Max_Loading * np.random.uniform(0.9, 1)
                    else:
                        x = 0.34 * Ix
                    y = 0.66 * Truck_Max_Loading -x
                else:
                    if(Ix > Truck_Max_Loading):
                        x = Truck_Max_Loading * np.random.uniform(0.9, 1)
                    else:
                        x = 0.34 * Ix
                    y = Truck_Max_Loading - x
            elif(Ix+Iy+Iz<=Truck_Max_Loading):
                    x = Ix / 3
                    y = Iy / 3
                    z = Iz / 3
            elif(Ix+Iy+Iz>Truck_Max_Loading and Ix+Iy+Iz<Truck_Max_Loading*3):
                if(Ix>=Truck_Max_Loading):
                    x = 0.66 * Truck_Max_Loading * np.random.uniform(0.9,1)
                    y = 0.66 * Truck_Max_Loading * np.random.uniform(0,0.1)
                else:
                    x = 0.34 * Ix
                    y = 0.66 * Truck_Max_Loading * np.random.uniform(0,1)
                z = 0.66 * Truck_Max_Loading - x - y
            elif(Ix+Iy+Iz >= Truck_Max_Loading*3):
                if(Ix>=Truck_Max_Loading):
                    x = Truck_Max_Loading * np.random.uniform(0.9,1)
                    y = Truck_Max_Loading * np.random.uniform(0,0.1)
                else:
                    x = 0.34 * Ix
                    y = Truck_Max_Loading * np.random.uniform(0, 1)
                z = Truck_Max_Loading - x - y
            x = int(round(x))
            y = int(round(y))
            z = int(round(z))
            if (Truck_Max_Loading<x+y+z or Ix<x or Iy<y or Iz<z or z<0):
                continue
            return x, y, z

# トラックの目的地の決め方
def SetTruckDestination(Start,t,df_Carried_Goods_tmp):
    # 行先が1
    if(Start==0):
        return 1

    # 行先が2 or 3
    elif(Start==1):
        Dx2_Expected = dx * Sum_Population2 * t
        Dy2_Expected = dy * Sum_Population2
        Dz2_Expected = dz * Sum_Population2 * t
        Dx3_Expected = dx * Sum_Population3 * t
        Dy3_Expected = dy * Sum_Population3
        Dz3_Expected = dz * Sum_Population3 * t
        Ix2_Expected = df_Carried_Goods_tmp.ix['t=' + str(t), 'To2(x)']
        Iy2_Expected = df_Carried_Goods_tmp.ix['t=' + str(t), 'To2(y)']
        Iz2_Expected = df_Carried_Goods_tmp.ix['t=' + str(t), 'To2(z)']
        Ix3_Expected = df_Carried_Goods_tmp.ix['t=' + str(t), 'To3(x)']
        Iy3_Expected = df_Carried_Goods_tmp.ix['t=' + str(t), 'To3(y)']
        Iz3_Expected = df_Carried_Goods_tmp.ix['t=' + str(t), 'To3(z)']
        Net_Demand2 = (Dx2_Expected - Ix2_Expected) + (Dy2_Expected - Iy2_Expected) + (Dz2_Expected - Iz2_Expected)
        Net_Demand3 = (Dx3_Expected - Ix3_Expected) + (Dy3_Expected - Iy3_Expected) + (Dz3_Expected - Iz3_Expected)
        if (Net_Demand2 >= Net_Demand3):
            Goal = 2
        else:
            Goal = 3
        return Goal

    # 行先が4,5,6 or 7,8,9
    elif(Start==2 or Start==3):
        if(Start==2):
            Goal_a = 4
            Goal_b = 5
            Goal_c = 6
        elif(Start==3):
            Goal_a = 7
            Goal_b = 8
            Goal_c = 9
        a_Dx_Expected = dx * Cap_Dictionary[str(Goal_a)] * t
        a_Dy_Expected = dy * Cap_Dictionary[str(Goal_a)]
        a_Dz_Expected = dz * Cap_Dictionary[str(Goal_a)] * t
        b_Dx_Expected = dx * Cap_Dictionary[str(Goal_b)] * t
        b_Dy_Expected = dy * Cap_Dictionary[str(Goal_b)]
        b_Dz_Expected = dz * Cap_Dictionary[str(Goal_b)] * t
        c_Dx_Expected = dx * Cap_Dictionary[str(Goal_c)] * t
        c_Dy_Expected = dy * Cap_Dictionary[str(Goal_c)]
        c_Dz_Expected = dz * Cap_Dictionary[str(Goal_c)] * t
        a_Ix_Expected = df_Carried_Goods_tmp.ix['t=' + str(t), 'To' + str(Goal_a) + '(x)']
        a_Iy_Expected = df_Carried_Goods_tmp.ix['t=' + str(t), 'To' + str(Goal_a) + '(y)']
        a_Iz_Expected = df_Carried_Goods_tmp.ix['t=' + str(t), 'To' + str(Goal_a) + '(z)']
        b_Ix_Expected = df_Carried_Goods_tmp.ix['t=' + str(t), 'To' + str(Goal_b) + '(x)']
        b_Iy_Expected = df_Carried_Goods_tmp.ix['t=' + str(t), 'To' + str(Goal_b) + '(y)']
        b_Iz_Expected = df_Carried_Goods_tmp.ix['t=' + str(t), 'To' + str(Goal_b) + '(z)']
        c_Ix_Expected = df_Carried_Goods_tmp.ix['t=' + str(t), 'To' + str(Goal_c) + '(x)']
        c_Iy_Expected = df_Carried_Goods_tmp.ix['t=' + str(t), 'To' + str(Goal_c) + '(y)']
        c_Iz_Expected = df_Carried_Goods_tmp.ix['t=' + str(t), 'To' + str(Goal_c) + '(z)']
        Net_Demand_a = (a_Dx_Expected - a_Ix_Expected) + (a_Dy_Expected - a_Iy_Expected) + (a_Dz_Expected - a_Iz_Expected)
        Net_Demand_b = (b_Dx_Expected - b_Ix_Expected) + (b_Dy_Expected - b_Iy_Expected) + (b_Dz_Expected - b_Iz_Expected)
        Net_Demand_c = (c_Dx_Expected - c_Ix_Expected) + (c_Dy_Expected - c_Iy_Expected) + (c_Dz_Expected - c_Iz_Expected)

        if(Net_Demand_a >= Net_Demand_b and Net_Demand_a >= Net_Demand_c):
            Goal = Goal_a
        elif(Net_Demand_b >= Net_Demand_a and Net_Demand_b >= Net_Demand_c):
            Goal = Goal_b
        else:
            Goal = Goal_c
        return Goal

# 乱数の初期化
np.random.seed(100)

# 距離
S01 = 1
S12 = 1
S13 = 1
S24 = 1
S25 = 1
S26 = 1
S37 = 1
S38 = 1
S39 = 1
S = [S01, S12, S13, S24, S25, S26, S37, S38, S39]

# 作業時間
LT0 = 1
LT1 = 1
LT2 = 1
LT3 = 1
LT = [LT0, LT1, LT2, LT3]
ULT1 = 1
ULT2 = 1
ULT3 = 1
ULT4 = 1
ULT5 = 1
ULT6 = 1
ULT7 = 1
ULT8 = 1
ULT9 = 1
ULT = [ULT1, ULT2, ULT3, ULT4, ULT5, ULT6, ULT7, ULT8, ULT9]

# 人口系の設定
Cap4 = 15000
Cap5 = 40000
Cap6 = 30000
Cap7 = 20000
Cap8 = 10000
Cap9 = 5000
Sum_Population2 = Cap4 + Cap5 + Cap6
Sum_Population3 = Cap7 + Cap8 + Cap9
Cap_Dictionary = {'4':Cap4, '5':Cap5, '6':Cap6, '7':Cap7, '8':Cap8, '9':Cap9}
Toatl_Population = 120000
Upper_mortality_rate = 0.05
Lower_mortality_rate = 0.01

# 支援物資「x」「y」「z」の一時間あたりの大人一人の需要量
dx = 0.1
dy = 1
dz = 0.00083

#　タイムステップ設定
t = 0
t_end = 24*10

# トラックの設定
Truck_Mini_Max_Loading = 350
Truck_1t_Max_Loading = 800
Truck_2t_Max_Loading = 2000
Truck_3t_Max_Loading = 3000
Truck_4t_Max_Loading = 4000
Truck_6t_Max_Loading = 6300
Truck_8t_Max_Loading = 8000
Truck_10t_Max_Loading = 10000
Quene = 0

# 被災者の避難状況を初期化。また、１時間後の需要と最終的な総需要の計算
P4, P5, P6, P7, P8, P9, Dead_Population = PopulationCaluculation()
Total_Supply_x, Total_Supply_y, Total_Supply_z = Total_Supply_calculation()

# トラックのデータフレーム
Initial_Truck = np.array([[0, 0, 0, LT0, Truck_10t_Max_Loading, 0, 0, 0],
                          [0, 0, 0, LT0, Truck_10t_Max_Loading, 0, 0, 0],
                          [0, 0, 0, LT0, Truck_10t_Max_Loading, 0, 0, 0],
                          [0, 0, 0, LT0, Truck_10t_Max_Loading, 0, 0, 0],
                          [0, 0, 0, LT0, Truck_10t_Max_Loading, 0, 0, 0],
                          [0, 0, 0, LT0, Truck_10t_Max_Loading, 0, 0, 0],
                          [0, 0, 0, LT0, Truck_10t_Max_Loading, 0, 0, 0],
                          [0, 0, 0, LT0, Truck_10t_Max_Loading, 0, 0, 0],
                          [0, 0, 0, LT0, Truck_10t_Max_Loading, 0, 0, 0],
                          [0, 0, 0, LT0, Truck_10t_Max_Loading, 0, 0, 0],
                          [1, 0, 0, LT0 + S01 + ULT1 + LT1, Truck_10t_Max_Loading, 0, 0, 0],
                          [1, 0, 0, LT0 + S01 + ULT1 + LT1, Truck_10t_Max_Loading, 0, 0, 0],
                          [1, 0, 0, LT0 + S01 + ULT1 + LT1, Truck_10t_Max_Loading, 0, 0, 0],
                          [1, 0, 0, LT0 + S01 + ULT1 + LT1, Truck_10t_Max_Loading, 0, 0, 0],
                          [1, 0, 0, LT0 + S01 + ULT1 + LT1, Truck_10t_Max_Loading, 0, 0, 0],
                          [1, 0, 0, LT0 + S01 + ULT1 + LT1, Truck_10t_Max_Loading, 0, 0, 0],
                          [2, 0, 0, LT0 + S01 + ULT1 + LT1 + S12 + ULT2 + LT2, Truck_10t_Max_Loading, 0, 0, 0],
                          [2, 0, 0, LT0 + S01 + ULT1 + LT1 + S12 + ULT2 + LT2, Truck_10t_Max_Loading, 0, 0, 0],
                          [2, 0, 0, LT0 + S01 + ULT1 + LT1 + S12 + ULT2 + LT2, Truck_10t_Max_Loading, 0, 0, 0],
                          [2, 0, 0, LT0 + S01 + ULT1 + LT1 + S12 + ULT2 + LT2, Truck_10t_Max_Loading, 0, 0, 0],
                          [3, 0, 0, LT0 + S01 + ULT1 + LT1 + S13 + ULT3 + LT3, Truck_10t_Max_Loading, 0, 0, 0],
                          [3, 0, 0, LT0 + S01 + ULT1 + LT1 + S13 + ULT3 + LT3, Truck_10t_Max_Loading, 0, 0, 0],
                          [3, 0, 0, LT0 + S01 + ULT1 + LT1 + S13 + ULT3 + LT3, Truck_10t_Max_Loading, 0, 0, 0],
                          [3, 0, 0, LT0 + S01 + ULT1 + LT1 + S13 + ULT3 + LT3, Truck_10t_Max_Loading, 0, 0, 0]])

df_Truck = pd.DataFrame(Initial_Truck.T, index=["Start","Goal","Phase","Work_Hours","Max_Truck_Loading","x","y","z"], columns=["truck" + str(i+1) for i in range(len(Initial_Truck))])

# 避難所の各支援物資に対する需要量
df_Demand = pd.DataFrame(Demandculation(P4, P5, P6, P7, P8, P9, np.zeros((1,18), dtype=np.int)),
                         index=['t=0'], columns=["4_Dx","4_Dy","4_Dz","5_Dx","5_Dy","5_Dz","6_Dx","6_Dy","6_Dz","7_Dx","7_Dy","7_Dz","8_Dx","8_Dy","8_Dz","9_Dx","9_Dy","9_Dz"])
dy = 0

# DC、Supplierの在庫量
df_Inventory = pd.DataFrame([[Total_Supply_x, Total_Supply_y, Total_Supply_z,0,0,0,0,0,0,0,0,0]],
                            index=['t=0'], columns=["0_Ix","0_Iy","0_Iz","1_Ix","1_Iy","1_Iz","2_Ix","2_Iy","2_Iz","3_Ix","3_Iy","3_Iz"])

# 各DC、避難所に運ばれた支援物資の総量
df_Carried_Goods = pd.DataFrame(np.zeros((1,24), dtype=np.int),
                        index=['t=0'],
                        columns=["To2(x)","To2(y)","To2(z)","To3(x)","To3(y)","To3(z)","To4(x)","To4(y)","To4(z)","To5(x)","To5(y)","To5(z)","To6(x)","To6(y)","To6(z)","To7(x)","To7(y)","To7(z)","To8(x)","To8(y)","To8(z)","To9(x)","To9(y)","To9(z)"])

# 初期値の出力
print("P4, P5, P6, P7, P8, P9, Dead_Population : %i, %i, %i, %i, %i, %i, %i" %(P4, P5, P6, P7, P8, P9, Dead_Population))

# シミュレーションの開始
t=1
while True:

    df_Demand_tmp = pd.DataFrame([Demandculation(P4, P5, P6, P7, P8, P9, df_Demand.ix['t='+str(t-1), :].as_matrix())],
                                 index=['t=' + str(t)],
                                 columns=["4_Dx", "4_Dy", "4_Dz", "5_Dx", "5_Dy", "5_Dz", "6_Dx", "6_Dy", "6_Dz",
                                          "7_Dx", "7_Dy", "7_Dz", "8_Dx", "8_Dy", "8_Dz", "9_Dx", "9_Dy", "9_Dz"])

    df_Inventory_tmp = pd.DataFrame([df_Inventory.ix['t='+str(t-1), :].as_matrix()], index=['t='+str(t)],
                                    columns=["0_Ix", "0_Iy", "0_Iz", "1_Ix", "1_Iy", "1_Iz", "2_Ix", "2_Iy", "2_Iz","3_Ix", "3_Iy", "3_Iz"])

    df_Carried_Goods_tmp = pd.DataFrame([df_Carried_Goods.ix['t='+str(t-1), :].as_matrix()], index=['t='+str(t)],
                                    columns=["To2(x)", "To2(y)", "To2(z)", "To3(x)", "To3(y)", "To3(z)", "To4(x)",
                                             "To4(y)", "To4(z)", "To5(x)", "To5(y)", "To5(z)", "To6(x)", "To6(y)",
                                             "To6(z)", "To7(x)", "To7(y)", "To7(z)", "To8(x)", "To8(y)", "To8(z)",
                                             "To9(x)", "To9(y)", "To9(z)"])

    for TruckSeries in df_Truck.iteritems():
        # Loading
        if (df_Truck.loc['Phase', TruckSeries[0]] == 0):
            df_Truck.loc['Work_Hours', TruckSeries[0]] -= 1

            # Loading終了　出発開始
            if(df_Truck.loc['Work_Hours', TruckSeries[0]] == 0):
                df_Truck.loc['Phase', TruckSeries[0]] = 1  #Phase変更
                df_Truck.loc['Work_Hours', TruckSeries[0]] = S[df_Truck.loc['Goal', TruckSeries[0]]-1]  #Work Hoursの設定

                # 積荷をトラックに載せる
                df_Truck.loc['x', TruckSeries[0]], df_Truck.loc['y', TruckSeries[0]], df_Truck.loc['z', TruckSeries[0]] \
                    = TruckLoading(df_Inventory_tmp.ix['t='+str(t), str(df_Truck.loc['Start', TruckSeries[0]]) + '_Ix'],
                                   df_Inventory_tmp.ix['t='+str(t), str(df_Truck.loc['Start', TruckSeries[0]]) + '_Iy'],
                                   df_Inventory_tmp.ix['t='+str(t), str(df_Truck.loc['Start', TruckSeries[0]]) + '_Iz'],
                                   df_Truck.loc["Max_Truck_Loading", TruckSeries[0]],
                                   df_Truck.loc['Start', TruckSeries[0]],t)

                # トラックに積み込んだ分をInventoryから引く
                df_Inventory_tmp.ix['t=' + str(t), str(df_Truck.loc['Start', TruckSeries[0]]) + '_Ix'] -= df_Truck.loc["x", TruckSeries[0]]
                df_Inventory_tmp.ix['t=' + str(t), str(df_Truck.loc['Start', TruckSeries[0]]) + '_Iy'] -= df_Truck.loc["y", TruckSeries[0]]
                df_Inventory_tmp.ix['t=' + str(t), str(df_Truck.loc['Start', TruckSeries[0]]) + '_Iz'] -= df_Truck.loc["z", TruckSeries[0]]

                # トラックの行き先を設定
                df_Truck.loc['Goal', TruckSeries[0]] = SetTruckDestination(df_Truck.loc['Start', TruckSeries[0]],t,df_Carried_Goods_tmp)

                # トラックに積み込んだ分を運ばれた支援物資量に加算する
                if(df_Truck.loc['Goal', TruckSeries[0]] != 1):
                    df_Carried_Goods_tmp.ix['t=' + str(t), 'To' + str(df_Truck.loc['Goal', TruckSeries[0]]) + '(x)'] += df_Truck.loc["x", TruckSeries[0]]
                    df_Carried_Goods_tmp.ix['t=' + str(t), 'To' + str(df_Truck.loc['Goal', TruckSeries[0]]) + '(y)'] += df_Truck.loc["y", TruckSeries[0]]
                    df_Carried_Goods_tmp.ix['t=' + str(t), 'To' + str(df_Truck.loc['Goal', TruckSeries[0]]) + '(z)'] += df_Truck.loc["z", TruckSeries[0]]

        # Travering to the goal point
        elif (df_Truck.loc['Phase', TruckSeries[0]] == 1):
            df_Truck.loc['Work_Hours', TruckSeries[0]] -= 1

            # Goal到着。　Unloading開始
            if (df_Truck.loc['Work_Hours', TruckSeries[0]] == 0):
                df_Truck.loc['Phase', TruckSeries[0]] = 2  #Phase変更
                df_Truck.loc['Work_Hours', TruckSeries[0]] = ULT[df_Truck.loc['Goal', TruckSeries[0]]-1]  #Work Hoursの設定

        # Unloacing
        elif (df_Truck.loc['Phase', TruckSeries[0]] == 2):
            df_Truck.loc['Work_Hours', TruckSeries[0]] -= 1

            # Unloading終了。　Startへ戻る
            if (df_Truck.loc['Work_Hours', TruckSeries[0]] == 0):
                df_Truck.loc['Phase', TruckSeries[0]] = 3  #Phase変更
                df_Truck.loc['Work_Hours', TruckSeries[0]] = S[df_Truck.loc['Goal', TruckSeries[0]] - 1]  #Work Hoursの設定

                # DCに到着していた場合
                if(df_Truck.loc['Goal', TruckSeries[0]] >= 1 and df_Truck.loc['Goal', TruckSeries[0]] <= 3):
                    df_Inventory_tmp.ix['t='+str(t), str(df_Truck.loc['Goal', TruckSeries[0]]) + '_Ix'] += df_Truck.loc["x", TruckSeries[0]]
                    df_Inventory_tmp.ix['t='+str(t), str(df_Truck.loc['Goal', TruckSeries[0]]) + '_Iy'] += df_Truck.loc["y", TruckSeries[0]]
                    df_Inventory_tmp.ix['t='+str(t), str(df_Truck.loc['Goal', TruckSeries[0]]) + '_Iz'] += df_Truck.loc["z", TruckSeries[0]]

                # Areaに到着していた場合
                elif(df_Truck.loc['Goal', TruckSeries[0]] >= 4 and df_Truck.loc['Goal', TruckSeries[0]] <= 9):
                    df_Demand_tmp.ix['t='+str(t), str(df_Truck.loc['Goal', TruckSeries[0]]) + '_Dx'] -= df_Truck.loc["x", TruckSeries[0]]
                    df_Demand_tmp.ix['t='+str(t), str(df_Truck.loc['Goal', TruckSeries[0]]) + '_Dy'] -= df_Truck.loc["y", TruckSeries[0]]
                    df_Demand_tmp.ix['t='+str(t), str(df_Truck.loc['Goal', TruckSeries[0]]) + '_Dz'] -= df_Truck.loc["z", TruckSeries[0]]

                df_Truck.loc['x', TruckSeries[0]], df_Truck.loc['y', TruckSeries[0]], df_Truck.loc['z', TruckSeries[0]]  = 0, 0, 0  #トラックの積荷を０にする

        # Travering to the start point
        elif (df_Truck.loc['Phase', TruckSeries[0]] == 3):
            df_Truck.loc['Work_Hours', TruckSeries[0]] -= 1

            # Startへ戻ってきた。
            if (df_Truck.loc['Work_Hours', TruckSeries[0]] == 0):
                df_Truck.loc['Phase', TruckSeries[0]] = 0  #Phase変更
                df_Truck.loc['Work_Hours', TruckSeries[0]] = LT[df_Truck.loc['Start', TruckSeries[0]]]  #Work Hoursの設定

    # 更新処理
    df_Demand = pd.concat([df_Demand, df_Demand_tmp])
    df_Inventory = pd.concat([df_Inventory, df_Inventory_tmp])
    df_Carried_Goods = pd.concat([df_Carried_Goods, df_Carried_Goods_tmp])

    # print("*******************************************************")
    print(t)
    # print(df_Truck)
    # print(df_Demand)
    print(df_Inventory)
    # print(df_Carried_Goods)

    t += 1
    if t > t_end:
        break

print(df_Inventory)
print(df_Demand)
print(df_Carried_Goods)

df_Demand.plot(figsize=(20,10))
plt.xlabel('Time (hour)')
plt.ylabel('Demand (kg)')
plt.show()