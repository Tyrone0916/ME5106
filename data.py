import math
import numpy as np

from scipy.io import loadmat


# 加载 .mat 文件
data_raw = loadmat('dataRaw.mat')
data_denoise = loadmat('dataDenoise.mat')
data_singletracks = loadmat('dataSingleTracks.mat')
data = {
    'A1': [350, 1200], 'A2': [400, 600], 'A3': [250, 600], 'A4': [250, 1000],
    'A5': [400, 1000], 'A7': [350, 700], 'A8': [200, 500], 'A9': [350, 500],
    'A10': [200, 400], 'A11': [200, 700], 'A12': [250, 500], 'A13': [400, 1200],
    'B1': [200, 200], 'B2': [300, 1000], 'B3': [150, 600], 'B4': [400, 800],
    'B5': [150, 500], 'B6': [300, 800], 'B7': [150, 400], 'B8': [300, 700],
    'B9': [400, 1600], 'B11': [150, 200], 'B13': [400, 2000]
}
# power_speed 激光功率和扫描速度


# hyperparameters
p_ref = 20 # u pa
vpp = 5 # mv/pa

# model parameters
num_epochs = 500




def calculate_moment_sound_pressure(V):
    # 计算瞬间声压 (p)
    # V 是测量的电压（单位：mV）
    # VPP 是麦克风的灵敏度（单位：mV/Pa）
    p = V / vpp
    return p

def calculate_rms_voltage(voltage_samples):
    # voltage_samples 是电压采样值的数组/列表
    n = len(voltage_samples)
    squared_sum = sum(v * v for v in voltage_samples)
    rms = math.sqrt(squared_sum / n)
    return rms

def calculate_spl(V):
    # 计算声压级 (SPL)
    # V 是测量的电压（单位：mV）
    # VPP 是麦克风的灵敏度（单位：mV/Pa） 5
    SPL = 20 * math.log10(V / vpp) + 94
    return SPL



# 显示加载的数据内容
# print(data_raw)
# print(data_denoise)
rms_result = {}
spl_result = {}
# print("数据结构信息：")
for key in data_singletracks.keys():
    if key != '__header__' and key != '__version__' and key != '__globals__' \
            and key != 'A6' and key != 'B10' and key != 'B12':
        # print(key)
        rms_result[key] = calculate_rms_voltage(data_singletracks[key])     # dict
        spl_result[key] = calculate_spl(rms_result[key])                    # dict

rms_result = np.array([
    (name, value) 
    for name, value in rms_result.items()
])
spl_result = np.array([
    (name, value) 
    for name, value in spl_result.items()
])

# print(isinstance(spl_result, np.ndarray))
print('spl_result',spl_result)
# print(isinstance(rms_result, np.ndarray))
print('rms_result',rms_result)

# 创建结构化数组
dt = np.dtype([
    ('name', 'U4'),              # 数据名称，最大4个字符
    ('laser_power', 'f4'),       # 激光功率，浮点数
    ('scan_speed', 'f4')         # 扫描速度，浮点数
])

# 转换数据为numpy数组
power_speed = np.array([
    (name, values[0], values[1]) 
    for name, values in data.items()
], dtype=dt)

# # 打印数组内容验证
# for row in  power_speed[:5]:
#     print(f"name: {row['name']}, laser_power: {row['laser_power']}, scan_speed: {row['scan_speed']}")