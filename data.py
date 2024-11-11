import math
import numpy as np

from scipy.io import loadmat


# 加载 .mat 文件
data_raw = loadmat('/home/tengyu/ME5106/mini_proj3/ME5106/dataRaw.mat')
data_denoise = loadmat('/home/tengyu/ME5106/mini_proj3/ME5106/dataDenoise.mat')
data_singletracks = loadmat('/home/tengyu/ME5106/mini_proj3/ME5106/dataSingleTracks.mat')
data = {
    'A1': [350, 1200], 'A2': [400, 600], 'A3': [250, 600], 'A4': [250, 1000],
    'A5': [400, 1000], 'A7': [350, 700], 'A8': [200, 500], 'A9': [350, 500],
    'A10': [200, 400], 'A11': [200, 700], 'A12': [250, 500], 'A13': [400, 1200],
    'B1': [200, 200], 'B2': [300, 1000], 'B3': [150, 600], 'B4': [400, 800],
    'B5': [150, 500], 'B6': [300, 800], 'B7': [150, 400], 'B8': [300, 700],
    'B9': [400, 1600], 'B11': [150, 200], 'B13': [400, 2000]
}
data_loc = {
    "A1": (0, 209.5, 307),
    "A2": (0, 232, 323),
    "A3": (0, 254.5, 340),
    "A4": (0, 277, 357),
    "A5": (0, 299.5, 375),
   # "A6": (22.5, 209.5, 308),
    "A7": (22.5, 232, 324),
    "A8": (22.5, 254.5, 340),
    "A9": (22.5, 277, 358),
    "A10": (22.5, 299.5, 375),
    "A11": (22.5, 322, 393),
    "A12": (22.5, 344.5, 412),
    "A13": (45, 209.5, 311),
    "B1": (45, 232, 326),
    "B2": (45, 254.5, 343),
    "B3": (45, 277, 360),
    "B4": (45, 299.5, 377),
    "B5": (45, 322, 395),
    "B6": (45, 344.5, 414),
    "B7": (67.5, 209.5, 315),
    "B8": (67.5, 232, 330),
    "B9": (67.5, 254.5, 346),
   # "B10": (67.5, 277, 363),
    "B11": (67.5, 299.5, 381),
   # "B12": (67.5, 322, 399),
    "B13": (67.5, 344.5, 417)
}


# features
# max
# average or  rms?
# loc




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
rms_max_result = {}
spl_rms_result = {}
spl_max_reuslt = {}

for key in data_singletracks.keys():
    if key != '__header__' and key != '__version__' and key != '__globals__' \
            and key != 'A6' and key != 'B10' and key != 'B12':
        # print(key)
        rms_result[key] = calculate_rms_voltage(data_singletracks[key])         # dict
        spl_rms_result[key] = calculate_spl(rms_result[key])                    # dict

        rms_max_result = max(data_singletracks[key])
        spl_max_reuslt[key] = calculate_spl(rms_max_result)



# rms_result = np.array([
#     (name, value) 
#     for name, value in rms_result.items()
# ])
spl_rms_result = np.array([
    (name, value) 
    for name, value in spl_rms_result.items()
])
spl_max_reuslt = np.array([
    (name, value) 
    for name, value in spl_max_reuslt.items()
])



# print(isinstance(spl_result, np.ndarray))
# print('spl_result',spl_result)
# print(isinstance(rms_result, np.ndarray))
# print('rms_result',rms_result)

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