"""
脚本功能： 计算鱼眼相机的畸变系数
"""

import openpyxl
import torch
import numpy as np
from scipy.optimize import curve_fit


class Get_Correct_Remap_Table():
    def __init__(self,
                 distortion_table_path,
                 focal_len=0.91):
        """
        数据准备：
        （1）theta
        （2）归一化平面上成像点位置
        """
        self.data = self.load_distortion_table(distortion_table_path)
        self.data[:, 0] = self.data[:, 0] / 180 * np.pi
        self.data[:, 1] = self.data[:, 1] / focal_len
        self.distor_para, _ = curve_fit(self.func, self.data[:, 0],
                                        self.data[:, 1])
        diff = np.abs(
            self.func(self.data[:, 0], *(self.distor_para)) - self.data[:, 1])
        print("error: ", np.mean(diff), np.sum(diff))

        r_d = self.func(self.data[:, 0], *(self.distor_para * 2.0))
        f_inverse_para, _ = curve_fit(self.func_inverse, r_d, self.data[:, 0])

        diff = np.abs(self.func_inverse(
            r_d, *(f_inverse_para)) - self.data[:, 0])
        print("inverse error: ", np.mean(diff), np.sum(diff))

    """
    读取畸变表中的数据：
    (1) theta: 入射角
    (2) r_d:   相机成像平面上成像点与成像中心的距离
    """

    def load_distortion_table(self, distortion_table_path):
        wb = openpyxl.load_workbook(distortion_table_path)
        wb = wb["Sheet1"]
        data = list(wb)
        data = [[x.value for x in row] for row in data]
        return np.array(data)

    def func(self, theta, k1, k2, k3, k4):
        return theta + k1 * theta**3 + k2 * theta**5 + k3 * theta**7 + k4 * theta**9

    def func_inverse(self, x, k1, k2, k3, k4):
        return x + k1 * x**2 + k2 * x**3 + k3 * x**4 + k4 * x**5


if __name__ == "__main__":
    """
        b26
        焦距: 0.91±5%mm
        像素大小: 3um*3um
    """

    config_dict = {}
    config_dict['distortion_table_path'] = "theta_rd_0414.xlsx"
    config_dict['focal_len'] = 0.91
    correct_method = Get_Correct_Remap_Table(**config_dict)
