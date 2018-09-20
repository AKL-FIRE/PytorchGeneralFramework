import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        '''此处编写模型组件初始化'''
        pass

    def forward(self, input):
        '''此处编写前向传播'''
        pass

def test():
    model = Model()
    print(model)
    x = torch.autograd.Variable(torch.randn(625, 5, 3, 1))
    y = model(x)
    print(y.size())


if __name__ == '__main__':
    test()