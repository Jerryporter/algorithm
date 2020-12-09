import numpy as np
import pandas as pd
import xlrd as xlrd




def excel_to_matrix(path):
    table = xlrd.open_workbook(path).sheets()[0]  # 获取第一个sheet表
    row = 10  # 行数
    col = 6  # 列数
    datamatrix = np.zeros((row, col))  # 生成一个nrows行ncols列，且元素均为0的初始矩阵
    for x in range(col + 1):
        if x == 0:
            continue
        cols = np.array(table.col_values(x)[1:11], dtype=np.float64)  # 把list转换为矩阵进行矩阵操作
        datamatrix[:, x - 1] = cols  # 按列把数据存进矩阵中
    # 数据归一化
    return datamatrix


def generate_4_GPSPositon(gpsX, gpsY, gpsZ):
    dict_position = {}
    for x in range(4):
        position = {x: np.array([gpsX[x], gpsY[x], gpsZ[x]], dtype=np.float64)}
        dict_position.update(position)
    return dict_position


def caculate_position_GPS(data,position_assume):
    gpsX = data[:4, 0]
    gpsY = data[:4, 1]
    gpsZ = data[:4, 2]
    pse = data[:4, 3]
    io = data[:4, 4]
    tr = data[:4, 5]
    while(True):
        #   (1)begin->calculate pse_true
        dict_position = generate_4_GPSPositon(gpsX, gpsY, gpsZ)
        pse_true = []
        for x in range(4):
            pse_true_x = np.linalg.norm(position_assume - dict_position.get(x))
            pse_true.append(pse_true_x)
        pse_true = np.array(pse_true)  # this is  distance between target and satellite,  which will be used to calculate
        #     (1) have finished
        # (2) begin->calculate A matrix
        A = np.ones((4, 4), dtype=np.float64)  # this is A matrix which will be used to calculate
        for x in range(4):
            tmp = position_assume - dict_position.get(x)
            tmp = tmp / pse_true[x]
            tmp = np.insert(tmp, 3, 1)
            A[x, :] = tmp
        #   (2) have finished
        # (3) begin->calculate free
        free_term = pse - pse_true
        calibration = io + tr
        free_term_calibrated = free_term - calibration  # this is free term which will be used to calculate
        # (3)have finished
        delta_positon = np.linalg.solve(A, free_term_calibrated)
        if(np.linalg.norm(delta_positon[:3])<=0.000001):
            print(position_assume)
            break
        position_assume=delta_positon[:3]+position_assume


# given_position = np.array([-1321303.501176, 530905.541405, 3237660.910000], dtype=np.float64)
given_position=np.array([2,54456,1241234])
datafile = "123.xlsx"
data = excel_to_matrix(datafile)
caculate_position_GPS(data,given_position)
