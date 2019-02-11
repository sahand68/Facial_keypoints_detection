## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 64,3)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv4 = nn.Conv2d(128, 256, 1)
        self.act1 = nn.ELU()
        self.act2 = nn.ELU()
        self.act3 = nn.ELU()
        self.act4 = nn.ELU()
        self.act5 = nn.ELU()
        self.act6 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout6 = nn.Dropout(0.6)
        self.maxpool = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(256*13*13,2048)
        self.fc2 = nn.Linear(2048,1024)
        self.fc3 = nn.Linear(1024, 136)
        self.fc4 = nn.Linear(136, 64)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)




    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.maxpool(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.maxpool(x)
        x = self.dropout3(x)
        x = self.conv4(x)
        x = self.act4(x)
        x = self.maxpool(x)
        x = self.dropout4(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.act5(x)
        x = self.dropout5(x)
        x = self.fc2(x)
        x = self.act6(x)
        x = self.dropout6(x)
        x = self.fc3(x)
        return x
