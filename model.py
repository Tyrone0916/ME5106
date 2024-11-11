import torch
import torch.nn as nn
import torch.optim as optim

from data import *

# 定义神经网络结构
class LaserProcessingNet(nn.Module):
    def __init__(self, spl_max_result, spl_rms_result, loc, power_speed):
        super(LaserProcessingNet, self).__init__()
        self.spl_max_result = spl_max_result
        self.spl_rms_result = spl_rms_result
        self.power_speed = power_speed
        self.loc = loc

        self.hidden1 = nn.Linear(5, 64)  # 输入层2个特征，隐藏层64个神经元
        self.hidden2 = nn.Linear(64, 32)  # 第二隐藏层32个神经元
        self.hidden3 = nn.Linear(32, 16)  # 第三隐藏层16个神经元
        self.output = nn.Linear(16, 2)  # 输出层2个神经元（激光功率，扫描速度）
    
    def forward(self, x):
        # 将输入特征（spl_max_result, spl_rms_result, power_speed）合并成一个向量
        # 假设spl_max_result和spl_rms_result是标量，power_speed是包含激光功率和扫描速度的向量
        # x = torch.cat((self.spl_max_result, self.spl_rms_result, self.power_speed), dim=1)  # 合并成一个向量
        
        # 通过神经网络各层进行前向传播
        x = torch.relu(self.hidden1(x))  # 第一层激活
        x = torch.relu(self.hidden2(x))  # 第二层激活
        x = torch.relu(self.hidden3(x))  # 第三层激活
        x = self.output(x)  # 输出层，返回激光功率和扫描速度

        return x




# # 输入数据（示例数据）
# X_train = torch.tensor([[80, 0.02, 0.5, 10, 100], 
#                         [85, 0.03, 0.6, 12, 110], 
#                         [90, 0.01, 0.4, 15, 120], 
#                         [75, 0.025, 0.45, 13, 95], 
#                         [100, 0.02, 0.55, 20, 130]], dtype=torch.float32)
# y_train = torch.tensor([[10, 100], 
#                         [12, 110], 
#                         [15, 120], 
#                         [13, 95], 
#                         [20, 130]], dtype=torch.float32)

# 训练模型
# num_epochs = 500
# for epoch in range(num_epochs):
#     model.train()
#     optimizer.zero_grad()
#     output = model(X_train)
#     loss = criterion(output, y_train)
#     loss.backward()
#     optimizer.step()
    
#     if (epoch + 1) % 50 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# # 测试模型（示例数据）
# model.eval()
# with torch.no_grad():
#     X_test = torch.tensor([[85, 0.022, 0.5, 11, 105], [95, 0.015, 0.45, 18, 125]], dtype=torch.float32)
#     y_pred = model(X_test)
#     print("Predicted Laser Power and Scanning Speed:", y_pred)
