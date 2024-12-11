import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# 使用SELayer的示例
if __name__ == '__main__':
    channel = 64  # 假设输入特征图的通道数为64
    se_layer = SELayer(channel)
    input_tensor = torch.randn(1, channel, 32, 32)  # 随机生成一个输入特征图
    output_tensor = se_layer(input_tensor)  # 通过SE模块处理输入特征图
    print(output_tensor.size())  # 打印输出特征图的尺寸