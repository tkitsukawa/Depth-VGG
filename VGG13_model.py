import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.block1_conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.block1_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.block1_batch = nn.BatchNorm2d(64)
        self.block1_pool = nn.MaxPool2d(2, stride=2)

        self.block2_conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.block2_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.block2_batch = nn.BatchNorm2d(128)
        self.block2_pool = nn.MaxPool2d(2, stride=2)

        self.block3_conv1 = nn.Conv2d(128, 256, 3, padding=1)
        self.block3_conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.block3_batch = nn.BatchNorm2d(256)
        self.block3_pool = nn.MaxPool2d(2, stride=2)

        self.block4_conv1 = nn.Conv2d(256, 512, 3, padding=1)
        self.block4_conv2 = nn.Conv2d(512, 512, 3, padding=1)
        self.block4_batch = nn.BatchNorm2d(512)
        self.block4_pool = nn.MaxPool2d(2, stride=2)

        self.block5_conv1 = nn.Conv2d(512, 256, 3, padding=1)
        self.block5_conv2 = nn.Conv2d(256, 1, 3, padding=1)
        self.block5_batch = nn.BatchNorm2d(1)
        self.block5_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = F.relu(self.block1_conv1(x))
        x = F.relu(self.block1_conv2(x))
        x = F.relu(self.block1_batch(x))
        x = self.block1_pool(x)

        x = F.relu(self.block2_conv1(x))
        x = F.relu(self.block2_conv2(x))
        x = F.relu(self.block2_batch(x))
        x = self.block2_pool(x)

        x = F.relu(self.block3_conv1(x))
        x = F.relu(self.block3_conv2(x))
        x = F.relu(self.block3_batch(x))
        x = self.block3_pool(x)

        x = F.relu(self.block4_conv1(x))
        x = F.relu(self.block4_conv2(x))
        x = F.relu(self.block4_batch(x))
        x = self.block4_pool(x)

        x = F.relu(self.block5_conv1(x))
        x = torch.sigmoid(self.block5_conv2(x))
        x = torch.sigmoid(self.block5_batch(x))
        # x = self.block4_pool(x)

        return x


class Miura_Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.block1_conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.block1_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.block1_batch = nn.BatchNorm2d(64)
        self.block1_pool = nn.MaxPool2d(2, stride=2)

        self.block2_conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.block2_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.block2_batch = nn.BatchNorm2d(128)
        self.block2_pool = nn.MaxPool2d(2, stride=2)

        self.block3_conv1 = nn.Conv2d(128, 256, 3, padding=1)
        self.block3_conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.block3_batch = nn.BatchNorm2d(256)
        self.block3_pool = nn.MaxPool2d(2, stride=2)

        self.block4_conv1 = nn.Conv2d(256, 512, 3, padding=1)
        self.block4_conv2 = nn.Conv2d(512, 512, 3, padding=1)
        self.block4_batch = nn.BatchNorm2d(512)
        self.block4_pool = nn.MaxPool2d(2, stride=2)

        self.block5_conv1 = nn.Conv2d(512, 256, 3, padding=1)
        self.block5_conv2 = nn.Conv2d(256, 1, 3, padding=1)
        self.block5_batch = nn.BatchNorm2d(1)
        self.block5_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = F.relu(self.block1_conv1(x))
        x = F.relu(self.block1_conv2(x))
        x = F.relu(self.block1_batch(x))
        x = self.block1_pool(x)

        x = F.relu(self.block2_conv1(x))
        x = F.relu(self.block2_conv2(x))
        x = F.relu(self.block2_batch(x))
        x = self.block2_pool(x)

        x = F.relu(self.block3_conv1(x))
        x = F.relu(self.block3_conv2(x))
        x = F.relu(self.block3_batch(x))
        x = self.block3_pool(x)

        x = F.relu(self.block4_conv1(x))
        x = F.relu(self.block4_conv2(x))
        x = F.relu(self.block4_batch(x))
        x = self.block4_pool(x)

        x = F.relu(self.block5_conv1(x))
        x = torch.sigmoid(self.block5_conv2(x))
        x = torch.sigmoid(self.block5_batch(x))
        # x = self.block4_pool(x)

        return x


# 以下，動作確認
if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Net()
    # model = model.to(device)
    # t = torch.rand([5, 3, 224, 224])
    # t = t.to(device)
    # output = model(t)
    # print(output.size())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net()
    model = model.to(device)
    t = torch.rand([10, 3, 224, 224])
    t = t.to(device)
    output = model(t)
    print(output.size())


