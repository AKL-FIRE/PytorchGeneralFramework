from model.model import Model
from torch.utils.data import DataLoader
from processor.feeder import Feeder
from torch.nn import functional as F
from train import Train
from predict import Predict


# 用于网络的训练
def train_main():

    LEARNING_RATE = 0.001
    EPOCH_NUM = 300
    BATCH_SIZE = 25

    train_dataset = Feeder(True, True)
    val_dataset = Feeder(False, True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Model()
    train = Train(model, LEARNING_RATE, EPOCH_NUM, BATCH_SIZE, train_loader, val_loader, len(train_dataset), len(val_loader))
    train.start()


# 用于网络的预测
def predict_main():
    BATCH_SIZE = 25

    test_dataset = Feeder(False, True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=25, shuffle=False)

    model = Model()
    predict = Predict(model, 'pt文件路径', test_loader)
    output = predict.predict_multi()
    output = F.softmax(output)  # 若网络最后无softmax（推荐）则加上这句。
    print(output)


if __name__ == '__main__':
    train_main()