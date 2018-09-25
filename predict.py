import torch


class Predict(object):

    def __init__(self, model, para_path, test_loader, use_gpu=True):
        '''
        :param model: 训练时使用的模型
        :param para_path: 训练得到的参数路径
        :param test_loader: 测试用数据读取器
        :param use_gpu: 是否使用gpu
        '''
        self.model = model
        self.model.load_state_dict(torch.load(para_path))  # 此处只读取数据，不读取模型
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.model = self.model.cuda()
        else:
            self.model = self.model.double()
        self.model.training = False
        self.model.eval()
        self.test_loader = test_loader

    # 通过loader进行对多数据进行分类
    def predict_multi(self):
        for input_data in self.test_loader:
            input_var = torch.autograd.Variable(input_data.cuda())
            output = self.model(input_var)
            return output

    # 对单一数据进行分类
    def predict_single(self, input_):
        input_var = torch.autograd.Variable(input_.cuda())
        output = self.model(input_var)
        return output
