import torch
import torch.nn as nn
import torch.optim as optim

# 定义神经网络结构
class LaserProcessingNet(nn.Module):
    def __init__(self):
        super(LaserProcessingNet, self).__init__()
        self.hidden1 = nn.Linear(2, 64)  # 输入层2个特征，隐藏层64个神经元
        self.hidden2 = nn.Linear(64, 32)  # 第二隐藏层32个神经元
        self.hidden3 = nn.Linear(32, 16)  # 第三隐藏层16个神经元
        self.output = nn.Linear(16, 2)  # 输出层2个神经元（激光功率，扫描速度）
    
    def forward(self, x):
        # 前向传播，通过激活函数 ReLU 连接各层
        x = torch.relu(self.hidden1(x))  # 第一层激活
        x = torch.relu(self.hidden2(x))  # 第二层激活
        x = torch.relu(self.hidden3(x))  # 第三层激活
        x = self.output(x)  # 输出层，返回激光功率和扫描速度
        return x

# 设置训练数据
# 输入数据（示例数据，包含平均频率和最大频率）
X_train = torch.tensor([
    [82.5, 2000], 
    [85.3, 1900], 
    [90.1, 2100], 
    [75.2, 1800], 
    [95.0, 2200]
], dtype=torch.float32)

# 输出数据（对应的激光功率和扫描速度）
y_train = torch.tensor([
    [350, 1200], 
    [400, 1000], 
    [450, 1400], 
    [300, 900], 
    [500, 1500]
], dtype=torch.float32)

# 创建模型实例
model = LaserProcessingNet()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 设置训练参数
num_epochs = 500

# 训练模型
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()  # 清零梯度
    output = model(X_train)  # 前向传播
    loss = criterion(output, y_train)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    
    # 每50个epoch打印一次损失
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型并比较预测值和真实值
model.eval()
with torch.no_grad():
    # 测试数据（输入平均频率和最大频率）
    X_test = torch.tensor([
        [83.0, 2050], 
        [87.5, 1950]
    ], dtype=torch.float32)

    # 真实输出（激光功率和扫描速度，用于比较）
    y_test = torch.tensor([
        [360, 1250], 
        [420, 1100]
    ], dtype=torch.float32)
    
    y_pred = model(X_test)  # 预测
    print("\n预测结果与真实值比较:")
    for i in range(len(y_test)):
        print(f"真实值: {y_test[i].numpy()}, 预测值: {y_pred[i].numpy()}")
