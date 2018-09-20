from model.model import Model
from torch.utils.data import DataLoader
from processor.feeder import Feeder
from train import Train


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


if __name__ == '__main__':
    train_main()