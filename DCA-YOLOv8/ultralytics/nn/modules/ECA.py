import torch
import torch.nn as nn

class ECALayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(channel, channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.gamma = gamma
        self.b = b

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.squeeze(3).transpose(1, 2)
        y = self.conv(y)
        y = y.transpose(1, 2).unsqueeze(3)
        y = self.sigmoid(y * self.gamma + self.b)
        return x * y

# 使用ECALayer的示例
# 假设channel为128

