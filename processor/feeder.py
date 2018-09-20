from torch.utils.data import Dataset
import torch

# operation
from processor.tools import *


class Feeder(Dataset):
    def __init__(self, is_train, use_gpu=True):
        self.use_gpu = use_gpu
        self.data = []
        self.label = []
        super(Feeder, self).__init__()
        if is_train:
            data = np.load('train_data.npy')
            label = np.load('train_label.npy')
        else:
            data = np.load('test_data.npy')
            label = np.load('test_label.npy')
        for i in range(len(data)):
            temp_data = np.zeros(('shape'))
            temp_data[:, :, :] = data['填对应维度']
            self.data.append(temp_data)
            self.label.append(int(label['填对应维度']))
        print('Data has been loaded!!!')

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        if self.use_gpu:
            return torch.from_numpy(data_numpy).float(), torch.LongTensor([label])
        else:
            return torch.from_numpy(data_numpy), torch.LongTensor([label])

    def __len__(self):
        return len(self.label)


def test():
    feeder = Feeder(True)


if __name__ == '__main__':
    test()