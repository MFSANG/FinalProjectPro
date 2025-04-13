import numpy as np
import math
import cmath
from math_tool import *

class mmWave_channel(object):
    """
    生成UMi开放场景下的毫米波信道（mmWave）
    输入：距离、角度、通信实体对象对
    输出：瞬时CSI（信道状态信息）
    """
    def __init__(self, transmitter, receiver, frequncy):
        """
        初始化毫米波信道类

        参数：
        transmitter: 发射方实体（来自entity.py）
        receiver: 接收方实体（来自entity.py）
        frequncy: 信道频率
        """
        self.channel_name = ''
        self.n = 0  # 路损指数
        self.sigma = 0  # 阴影衰落标准差
        self.transmitter = transmitter
        self.receiver = receiver
        self.channel_type = self.init_type()  # 初始化信道类型

        self.frequncy = frequncy
        # 初始化并更新路径损耗
        self.path_loss_normal = self.get_channel_path_loss()  # 普通值
        self.path_loss_dB = normal_to_dB(self.path_loss_normal)  # dB值
        # 初始化并更新CSI矩阵
        self.channel_matrix = self.get_estimated_channel_matrix()

    def init_type(self):
        """
        初始化信道类型，根据发射端和接收端的类型决定参数设置
        """
        channel_type = self.transmitter.type + '_' + self.receiver.type
        if channel_type == 'UAV_RIS' or channel_type == 'RIS_UAV':
            self.n = 2.2
            self.sigma = 3
            self.channel_name = 'H_UR'
        elif channel_type == 'UAV_user' or channel_type == 'UAV_attacker':
            self.n = 3.5
            self.sigma = 3
            if channel_type == 'UAV_user':
                self.channel_name = 'h_U_k,' + str(self.transmitter.index)
            elif channel_type == 'UAV_attacker':
                self.channel_name = 'h_U_p,' + str(self.transmitter.index)
        elif channel_type == 'user_UAV' or channel_type == 'attacker_UAV':
            self.n = 3.5
            self.sigma = 3
            if channel_type == 'user_UAV':
                self.channel_name = 'h_U_k,' + str(self.transmitter.index)
            elif channel_type == 'attacker_UAV':
                self.channel_name = 'h_U_p,' + str(self.transmitter.index)
        elif channel_type == 'RIS_user' or channel_type == 'RIS_attacker':
            self.n = 2.8
            self.sigma = 3
            if channel_type == 'RIS_user':
                self.channel_name = 'h_R_k,' + str(self.transmitter.index)
            elif channel_type == 'RIS_attacker':
                self.channel_name = 'h_R_p,' + str(self.transmitter.index)
        elif channel_type == 'user_RIS' or channel_type == 'attacker_RIS':
            self.n = 2.8
            self.sigma = 3
            if channel_type == 'user_RIS':
                self.channel_name = 'h_R_k,' + str(self.transmitter.index)
            elif channel_type == 'attacker_RIS':
                self.channel_name = 'h_R_p,' + str(self.transmitter.index)
        return channel_type

    def get_channel_path_loss(self):
        """
        计算路径损耗，包括阴影衰落（以普通值返回）
        """
        distance = np.linalg.norm(self.transmitter.coordinate - self.receiver.coordinate)
        # 自由空间路径损耗计算公式
        PL = -20 * math.log10(4*math.pi/(3e8/self.frequncy)) - 10*self.n*math.log10(distance)
        shadow_loss = np.random.normal() * self.sigma
        # 返回dB转普通值（可以注释阴影衰落或保留）
        return dB_to_normal(PL)

    def get_estimated_channel_matrix(self):
        """
        初始化并更新信道矩阵（CSI）
        """
        # 发射端/接收端天线数量
        N_t = self.transmitter.ant_num
        N_r = self.receiver.ant_num

        # 初始单位矩阵（复数）
        channel_matrix = np.mat(np.ones(shape=(N_r, N_t), dtype=complex), dtype=complex)

        # 计算接收端在发射端坐标系下的坐标
        r_under_t_car_coor = get_coor_ref(
            self.transmitter.coor_sys,
            self.receiver.coordinate - self.transmitter.coordinate
        )
        # 转换为球坐标（r, θ, φ）
        r_t_r, r_t_theta, r_t_fai = cartesian_coordinate_to_spherical_coordinate(r_under_t_car_coor)

        # 计算发射端在接收端坐标系下的坐标
        t_under_r_car_coor = get_coor_ref(
            [-self.receiver.coor_sys[0], self.receiver.coor_sys[1], -self.receiver.coor_sys[2]],
            self.transmitter.coordinate - self.receiver.coordinate
        )
        # 转换为球坐标
        t_r_r, t_r_theta, t_r_fai = cartesian_coordinate_to_spherical_coordinate(t_under_r_car_coor)

        # 计算天线方向响应（阵列响应）
        t_array_response = self.generate_array_response(self.transmitter, r_t_theta, r_t_fai)
        r_array_response = self.generate_array_response(self.receiver, t_r_theta, t_r_fai)
        array_response_product = r_array_response * t_array_response.H  # H为共轭转置

        # 获取路径损耗
        PL = self.path_loss_normal
        # 计算相位偏移
        LOS_fai = 2 * math.pi * self.frequncy * np.linalg.norm(self.transmitter.coordinate - self.receiver.coordinate) / 3e8
        # 计算LOS条件下的信道矩阵
        channel_matrix = cmath.exp(1j * LOS_fai) * math.sqrt(PL) * array_response_product

        return channel_matrix

    def generate_array_response(self, transceiver, theta, fai):
        """
        生成阵列响应向量，支持UPA、ULA、单天线三种天线类型
        """
        ant_type = transceiver.ant_type
        ant_num = transceiver.ant_num

        if ant_type == 'UPA':
            row_num = int(math.sqrt(ant_num))
            Planar_response = np.mat(np.ones(shape=(ant_num, 1)), dtype=complex)
            for i in range(row_num):
                for j in range(row_num):
                    Planar_response[j + i * row_num, 0] = cmath.exp(1j * (
                        math.sin(theta) * math.cos(fai) * i * math.pi + math.sin(theta) * math.sin(fai)
                    ))
            return Planar_response

        elif ant_type == 'ULA':
            Linear_response = np.mat(np.ones(shape=(ant_num, 1)), dtype=complex)
            for i in range(ant_num):
                Linear_response[i, 0] = cmath.exp(1j * math.sin(theta) * math.cos(fai) * i * math.pi)
            return Linear_response

        elif ant_type == 'single':
            return np.mat(np.array([1]))  # 单天线返回常量
        else:
            return False

    def update_CSI(self):
        """
        更新路径损耗和信道矩阵（用于移动通信场景中重新计算）
        """
        self.path_loss_normal = self.get_channel_path_loss()
        self.path_loss_dB = normal_to_dB(self.path_loss_normal)
        self.channel_matrix = self.get_estimated_channel_matrix()
