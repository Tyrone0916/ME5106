from data import *
from model import LaserProcessingNet

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def train(spl_max_result, spl_rms_result, loc, power_speed, epochs=num_epochs):
    # 数据准备
    # 确保只使用数值数据，去掉字符串
    spl_values = np.array([x[1] for x in spl_max_result], dtype=np.float32)
    rms_values = np.array([x[1] for x in spl_rms_result], dtype=np.float32)
    power_values = power_speed['laser_power'].astype(np.float32)
    speed_values = power_speed['scan_speed'].astype(np.float32)
    loc = np.array(list(data_loc.values()))
    # loc = data_loc.astype(np.float32)
    
    # 将输入特征组合在一起
    X = torch.tensor(np.column_stack((spl_values, rms_values, loc)), dtype=torch.float32)
    # print('X',X)
    # print('X.shape',X.shape)
    y = torch.tensor(np.column_stack((power_values, speed_values)), dtype=torch.float32)
    # print("Final X shape:", X.shape) # [23,2]
    # print("Final y shape:", y.shape) # [23,2]
    
    # 创建数据集和数据加载器
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    # 创建模型实例
    model = LaserProcessingNet(spl_max_result, spl_rms_result, loc, power_speed)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            # 前向传播
            outputs = model.forward(batch_X)
            loss = criterion(outputs, batch_y)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 每50个epoch打印一次损失
        if (epoch + 1) % 50 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
            
            # 简单的验证
            model.eval()
            with torch.no_grad():
                test_output = model(X)
                test_loss = criterion(test_output, y)
                print(f'Validation Loss: {test_loss:.4f}')
                
                # 打印一些预测结果
                print("预测样本:")
                for i in range(min(3, len(test_output))):
                    print(f'真实值: {y[i].numpy()}, 预测值: {test_output[i].numpy()}')
    
    return model


if __name__ == "__main__":
    trained_model = train(spl_max_reuslt, spl_rms_result, data_loc, power_speed)
    # 保存模型
    torch.save(trained_model.state_dict(), 'laser_model.pth')
    # 测试模型
    trained_model.eval()
    with torch.no_grad():
        # 准备测试数据
        test_X = torch.tensor([[82.6, 1.355, 22.5, 299.5, 375]], dtype=torch.float32) #"A10"
        predictions = trained_model(test_X)
        print("\n测试结果:")
        print("预测的激光功率和扫描速度:", predictions.numpy())
        print("真实的激光功率和扫描速度:200， 400")
# 24% 73%
