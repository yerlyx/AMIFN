import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class myResnet(nn.Module):
    def __init__(self, resnet, if_fine_tune, device):
        super(myResnet, self).__init__()
        self.resnet = resnet
        self.if_fine_tune = if_fine_tune
        self.device = device

    def forward(self, x, att_size=7):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        fc = x.mean(3).mean(2)
        att = F.adaptive_avg_pool2d(x,[att_size,att_size])

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)

        if not self.if_fine_tune:
            
            x= Variable(x.data)
            fc = Variable(fc.data)
            att = Variable(att.data)

        return x, fc, att
    #x是通过全局池化（平均池化）操作后得到的特征向量。
    # 它通常用于图像分类任务或作为其他模型的输入。
    # fc是通过空间维度上的平均池化操作得到的特征矩阵。
    # 它捕捉了图像中每个通道的平均特征表示，并且在一些空间注意力场景中可能会用到。
    # att是通过自适应平均池化操作得到的空间注意力矩阵。
    # 它可以捕获图像中每个通道在不同空间位置的重要性，并且通常用于图像生成任务或在注意力机制中使用。


