import numpy as np
import scipy.io
import pandas as pd
import os
import time, csv

class DataManager(object):
    """
    仿真结果读写与管理类
    使用前需在当前路径下创建文件夹 './data'，并放置包含实体位置的 'init_location.xlsx' 文件。
    """

    def __init__(self, store_list=['beamforming_matrix', 'reflecting_coefficient', 'UAV_state', 'user_capacity'], file_path='./data', store_path='./data/storage', project_name=None):
        """
        初始化数据管理器

        参数：
        store_list：需要存储的字段列表（对应不同仿真结果）
        file_path：初始化位置文件路径
        store_path：结果保存路径
        project_name：项目名（用于命名保存文件夹）
        """
        self.store_list = store_list
        self.init_data_file = file_path + '/init_location.xlsx'  # 实体初始位置数据

        # 根据是否有项目名决定保存路径
        if project_name is None:
            # 自动生成时间戳作为文件夹名
            self.time_stemp = time.strftime('/%Y-%m-%d %H_%M_%S', time.localtime(time.time()))
            self.store_path = store_path + self.time_stemp
        else:
            # 如果项目重名，自动添加后缀避免重复
            for i in range(1, 100, 1):
                if i == 1:
                    dir_name = store_path + '/' + project_name
                else:
                    dir_name = store_path + '/' + project_name + f'_{i}'
                if not os.path.isdir(dir_name):
                    self.store_path = dir_name
                    break

        os.makedirs(self.store_path)  # 创建保存目录
        self.simulation_result_dic = {}  # 存储结果的字典
        self.init_format()  # 初始化存储格式

    def save_file(self, episode_cnt=10):
        """
        保存当前回合的仿真数据：
        - 记录该回合的step数量
        - 保存为.mat文件
        - 清空结果字典，准备下次记录
        """
        # 保存每回合step数
        with open(self.store_path + "/step_num_per_episode.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([len(list(self.simulation_result_dic.values())[0])])  # 任意字段长度即可

        # 保存仿真结果为MATLAB文件格式
        scipy.io.savemat(
            self.store_path + '/simulation_result_ep_' + str(episode_cnt) + '.mat',
            {'result_' + str(episode_cnt): self.simulation_result_dic}
        )

        # 清空数据，为下一回合准备
        self.simulation_result_dic = {}
        self.init_format()

    def save_meta_data(self, meta_dic):
        """
        保存系统或Agent的元数据（参数设置、结构信息等）
        """
        scipy.io.savemat(self.store_path + '/meta_data.mat', {'meta_data': meta_dic})

    def init_format(self):
        """
        初始化用于存储的字段结构（每次仿真前调用）
        """
        for store_item in self.store_list:
            self.simulation_result_dic.update({store_item: []})

    def read_init_location(self, entity_type='user', index=0):
        """
        读取指定实体的初始位置信息

        参数：
        entity_type：实体类型（user, attacker, RIS, UAV, RIS_norm_vec等）
        index：实体在表中的索引位置

        返回：
        [x, y, z] 坐标组成的 numpy 数组
        """
        if entity_type == 'user' or 'attacker' or 'RIS' or 'RIS_norm_vec' or 'UAV':
            return np.array([
                pd.read_excel(self.init_data_file, sheet_name=entity_type)['x'][index],
                pd.read_excel(self.init_data_file, sheet_name=entity_type)['y'][index],
                pd.read_excel(self.init_data_file, sheet_name=entity_type)['z'][index]
            ])
        else:
            return None

    def store_data(self, row_data, value_name):
        """
        存储单步仿真数据到结果字典中

        参数：
        row_data：当前数据（如数组、数值等）
        value_name：存储键（需包含在 store_list 中）
        """
        self.simulation_result_dic[value_name].append(row_data)
