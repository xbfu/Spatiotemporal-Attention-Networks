import torch
from torch import nn
import numpy as np
import pandas as pd
import torch.cuda as cuda
import torch.optim.lr_scheduler as lr_scheduler
from torch import optim
from torch.utils.data import DataLoader, Dataset
from model import Model

# Hyper Parameters
BATCH = 300
EPOCHS = 40
INPUT_SIZE = 6
LR = 0.01
d_model = 512
heads = 8
HIDDEN_SIZE = 32
h_state = None
TIME_STEP = 12
STEPS = 1
DEVICE = torch.device('cuda' if cuda.is_available() else 'cpu')


def generator(seq):
    size = 16030
    seq = np.array(seq)
    data = []
    for i in range(size - 30):
        data.append(np.reshape(seq[i:i + TIME_STEP + 24, 3], newshape=[1, -1]))
    data = np.reshape(data, newshape=[-1, TIME_STEP + 24])
    return np.array(data, dtype=np.float32)


class DealDateset(Dataset):
    def __init__(self):
        x1 = torch.from_numpy(np.reshape(train_data[:, 0:TIME_STEP], newshape=(-1, TIME_STEP, 1)))
        x2 = torch.from_numpy(np.reshape(train_dataA[:, 0:TIME_STEP], newshape=(-1, TIME_STEP, 1)))
        x3 = torch.from_numpy(np.reshape(train_dataB[:, 0:TIME_STEP], newshape=(-1, TIME_STEP, 1)))
        x4 = torch.from_numpy(np.reshape(train_dataC[:, 0:TIME_STEP], newshape=(-1, TIME_STEP, 1)))
        x5 = torch.from_numpy(np.reshape(train_dataD[:, 0:TIME_STEP], newshape=(-1, TIME_STEP, 1)))
        x6 = torch.from_numpy(np.reshape(train_dataE[:, 0:TIME_STEP], newshape=(-1, TIME_STEP, 1)))
        self.x = torch.cat((x1, x2, x3, x4, x5, x6), dim=2)

        self.y = torch.from_numpy(np.reshape(train_data[:, TIME_STEP + STEPS - 1], newshape=(-1, 1, 1)))
        self.len = train_data.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


def prepare_data(path):
    table = pd.read_csv(path, sep=",", header=None, skiprows=1)
    data = generator(table)
    data = data / 1200
    train_data = data[0:10000]
    test_data = data[11000:11130]
    return train_data, test_data


train_data, test_data = prepare_data(path="/home/owen/SITE_00193.CSV")
train_dataA, test_dataA = prepare_data(path="/home/owen/SITE_00173.CSV")
train_dataB, test_dataB = prepare_data(path="/home/owen/SITE_00215.CSV")
train_dataC, test_dataC = prepare_data(path="/home/owen/SITE_00365.CSV")
train_dataD, test_dataD = prepare_data(path="/home/owen/SITE_00446.CSV")
train_dataE, test_dataE = prepare_data(path="/home/owen/SITE_00797.CSV")

x1 = torch.from_numpy(np.reshape(test_data[:, 0:TIME_STEP], newshape=(-1, TIME_STEP, 1)))
x2 = torch.from_numpy(np.reshape(test_dataA[:, 0:TIME_STEP], newshape=(-1, TIME_STEP, 1)))
x3 = torch.from_numpy(np.reshape(test_dataB[:, 0:TIME_STEP], newshape=(-1, TIME_STEP, 1)))
x4 = torch.from_numpy(np.reshape(test_dataC[:, 0:TIME_STEP], newshape=(-1, TIME_STEP, 1)))
x5 = torch.from_numpy(np.reshape(test_dataD[:, 0:TIME_STEP], newshape=(-1, TIME_STEP, 1)))
x6 = torch.from_numpy(np.reshape(test_dataE[:, 0:TIME_STEP], newshape=(-1, TIME_STEP, 1)))
test_x = torch.cat((x1, x2, x3, x4, x5, x6), dim=2)

test_y = torch.from_numpy(np.reshape(test_data[:, TIME_STEP + STEPS - 1], newshape=(-1, 1, 1)))
test_x = test_x.to(DEVICE)
test_y = test_y.squeeze().to(DEVICE)

dataset = DealDateset()
loader = DataLoader(dataset=dataset, batch_size=BATCH, shuffle=True)
print('Read data: finished')

model = Model().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for step in range(EPOCHS):
    scheduler.step()
    for i, (train_x, train_y) in enumerate(loader):
        pred = model(train_x.to(DEVICE))
        loss = criterion(pred.squeeze(), train_y.squeeze().to(DEVICE))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    rmse_train = torch.sqrt(loss)
    y_ = model(test_x.to(DEVICE))
    rmse_test = criterion(y_.squeeze(), test_y.squeeze().to(DEVICE))
    rmse_test = torch.sqrt(rmse_test)
    loss_test = torch.mean(torch.abs(1 - y_.squeeze() / test_y.squeeze().to(DEVICE)))
    print('Epoch:%3d' % (step + 1),
          '| Train RMSE: %.5f' % rmse_train,
          '| Test RMSE : %.5f' % rmse_test)
