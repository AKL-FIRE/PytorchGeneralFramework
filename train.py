import torch
import torch.nn as nn
import torch.utils.data.dataloader
import numpy as np


class Train(object):

    def __init__(self, model, base_lr, epoch_mun, batch_size, train_loader, val_loader=None, train_dataset_len=None, val_dataset_len=None, use_gpu=True):
        '''
        :param model: 选择的模型
        :param base_lr: 基础学习率
        :param epoch_mun: 迭代次數
        :param batch_size:　每批数据大小
        :param train_loader:　训练数据读取器
        :param val_loader:　验证数据读取器
        :param train_dataset_len:　训练数据大小
        :param val_dataset_len:　验证数据大小
        :param use_gpu:　是否使用ｇｐｕ
        '''
        self.model = model
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.model = self.model.cuda()
        else:
            self.model = self.model.double()
        self.base_lr = base_lr
        # 此处选择优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.base_lr, weight_decay=0.0001)
        self.epoch_num = epoch_mun
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_cross = nn.CrossEntropyLoss()
        self.train_len = train_dataset_len
        self.val_len = val_dataset_len
        self.output_model_path = './result/trained_model/model_'
        self.output_val_path = './result/txt/result_'
        temp_str = '{}_{}_{}_{}'.format('optimizer_name', self.base_lr, self.epoch_num, self.batch_size)
        self.output_model_path = self.output_model_path + temp_str + '.pt'
        self.output_val_path = self.output_val_path + temp_str + '.txt'

    def start(self):
        np.set_printoptions(precision=4, threshold=np.nan)
        self.model.train()
        # total_acc为Ｔｒｕｅ表示使用总正确率，
        total_acc = False
        for epoch in range(self.epoch_num):
            for step, (input, target) in enumerate(self.train_loader):

                if input.size(0) == self.batch_size:
                    if self.use_gpu:
                        input_var = torch.autograd.Variable(input.cuda())
                        target_var = torch.autograd.Variable(target.cuda())
                    else:
                        input_var = torch.autograd.Variable(input)
                        target_var = torch.autograd.Variable(target)

                    # 前向传播
                    output = self.model(input_var)

                    # 计算损失
                    loss = self.loss_cross(output, target_var)

                    # 反向传播
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # 训练输出
                    if step % 50 == 0:
                        print('epoch : ({} / {}) , step : ({} / {}), loss : {}'.format(epoch, self.epoch_num, step, self.train_len // self.batch_size, loss.data[0]))

            # 调整学习率
            self.adjust_learning_rate(self.optimizer, epoch)

            if epoch % 50 == 0 and total_acc:
                print('#########################TEST_START##########################')
                self.model.eval()
                eval_loss = 0
                eval_acc = 0
                for i, (input, target) in enumerate(self.val_loader):
                    if input.size(0) == self.batch_size:
                        if self.use_gpu:
                            input_var = torch.autograd.Variable(input.cuda(), volatile=True)
                            target_var = torch.autograd.Variable(target.cuda(), volatile=True)
                        else:
                            input_var = torch.autograd.Variable(input, volatile=True)
                            target_var = torch.autograd.Variable(target, volatile=True)

                        output_vali = self.model(input_var)
                        loss_val, accuracy, raw_loss, raw_acc = self.accuracy(output_vali, target_var, self.loss_cross)
                        eval_loss += raw_loss
                        eval_acc += raw_acc
                print('The whole loss :{} , accuracy : {}'.format(eval_loss / self.val_len, eval_acc / self.val_len))
                with open(self.output_val_path, 'a') as f:
                    f.write('epoch : ({} / {}) , loss : {} , accuracy : {} \n'.format(epoch, self.epoch_num, eval_loss / self.val_len, eval_acc / self.val_len))
                print('#########################TEST_END##########################')
                self.model.train()

            if epoch % 50 == 0 and not total_acc:
                print('#########################TEST_START##########################')
                self.model.eval()
                eval_loss = 0
                eval_acc = np.zeros(('number of class'), np.int32)
                eval_total = np.zeros(('number of class'), np.int32)
                for i, (input, target) in enumerate(self.val_loader):
                    if input.size(0) == self.batch_size:
                        if self.use_gpu:
                            input_var = torch.autograd.Variable(input.cuda(), volatile=True)
                            target_var = torch.autograd.Variable(target.cuda(), volatile=True)
                        else:
                            input_var = torch.autograd.Variable(input, volatile=True)
                            target_var = torch.autograd.Variable(target, volatile=True)

                        output_vali = self.model(input_var)
                        raw_acc, raw_loss, raw_total = self.accuracy_distinct(output_vali, target_var, self.loss_cross)
                        eval_loss += raw_loss
                        eval_acc += raw_acc
                        eval_total += raw_total
                print('The whole loss :{} , accuracy : {}'.format(eval_loss / self.val_len, eval_acc / eval_total))
                with open(self.output_val_path, 'a') as f:
                    f.write('epoch : ({} / {}) , loss : {} , accuracy : {} \n'.format(epoch, self.epoch_num, eval_loss / self.val_len, eval_acc / eval_total))
                print('#########################TEST_END##########################')
                self.model.train()
        torch.save(self.model.state_dict(), self.output_model_path)

    # 仅仅计算总分类正确率
    @staticmethod
    def accuracy(output_vali, target, criterion):
        loss = criterion(output_vali, target)
        eval_loss = loss.data[0] * target.size(0)
        _, pred = torch.max(output_vali, 1)
        num_correct = (pred == target).sum()
        eval_acc = num_correct.data[0]
        return eval_loss / len(target), eval_acc / len(target), eval_loss, eval_acc

    # 分类计算各类正确率
    @staticmethod
    def accuracy_distinct(output_vali, target, criterion):
        loss = criterion(output_vali, target)
        eval_loss = loss.data[0] * target.size(0)
        _, pred = torch.max(output_vali, 1)
        num_correct = np.zeros((output_vali.size(1)), np.int32)
        num_total = np.zeros((output_vali.size(1)), np.int32)
        for i in range(25):
            num_correct[target.data[i]] += (pred.data[i] == target.data[i])
            num_total[target.data[i]] += 1
        return num_correct, eval_loss, num_total

    # 根据迭代epoch进行学习率衰减
    def adjust_learning_rate(self, optimizer, epoch):
        if epoch % 1 == 0:
            print('################## Adjust Learning rate ###############')
            print('New learning rate is : {}'.format(self.base_lr * 0.99))
            print('############### Adjust Learning rate end ##############')
            self.base_lr *= 0.99
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.base_lr

    # 指数学习率衰减
    def adjust_learning_rate_exp(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
        if epoch % 10 == 0:
            print('################## Adjust Learning rate ###############')
            print('New learning rate is : {}'.format(self.base_lr * (0.1 ** (epoch // 10))))
        lr = self.base_lr * (0.1 ** (epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
