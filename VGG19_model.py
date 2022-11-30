import torch
import torch.nn as nn
import torch.nn.functional as f


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Block1 畳み込み層×2,BatchNormalization,Maxプーリング層
        # nn.Conv2d(入力チャンネル数,出力チャンネル数,カーネルサイズ,パディングの量)
        self.block1_conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.block1_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.block1_batch = nn.BatchNorm2d(64)
        self.block1_pool = nn.MaxPool2d(2, stride=2)

        # Block2 畳み込み層×2,BatchNormalization,Maxプーリング層
        self.block2_conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.block2_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.block2_batch = nn.BatchNorm2d(128)
        self.block2_pool = nn.MaxPool2d(2, stride=2)

        # Block3 畳み込み層×4,BatchNormalization,Maxプーリング層
        self.block3_conv1 = nn.Conv2d(128, 256, 3, padding=1)
        self.block3_conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.block3_conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.block3_conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.block3_batch = nn.BatchNorm2d(256)
        self.block3_pool = nn.MaxPool2d(2, stride=2)

        # Block4 畳み込み層×4,BatchNormalization,Maxプーリング層
        self.block4_conv1 = nn.Conv2d(256, 512, 3, padding=1)
        self.block4_conv2 = nn.Conv2d(512, 512, 3, padding=1)
        self.block4_conv3 = nn.Conv2d(512, 512, 3, padding=1)
        self.block4_conv4 = nn.Conv2d(512, 512, 3, padding=1)
        self.block4_batch = nn.BatchNorm2d(512)
        self.block4_pool = nn.MaxPool2d(2, stride=2)

        # Block5 畳み込み層×4,BatchNormalization,Maxプーリング層
        self.block5_conv1 = nn.Conv2d(512, 512, 3, padding=1)
        self.block5_conv2 = nn.Conv2d(512, 512, 3, padding=1)
        self.block5_conv3 = nn.Conv2d(512, 64, 3, padding=1)
        self.block5_conv4 = nn.Conv2d(64, 1, 3, padding=1)
        self.block5_batch = nn.BatchNorm2d(1)
        # self.block5_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        # Block1
        x = f.relu(self.block1_conv1(x))
        x = f.relu(self.block1_conv2(x))
        x = f.relu(self.block1_batch(x))
        x = self.block1_pool(x)

        # Block2
        x = f.relu(self.block2_conv1(x))
        x = f.relu(self.block2_conv2(x))
        x = f.relu(self.block2_batch(x))
        x = self.block2_pool(x)

        # Block3
        x = f.relu(self.block3_conv1(x))
        x = f.relu(self.block3_conv2(x))
        x = f.relu(self.block3_conv3(x))
        x = f.relu(self.block3_conv4(x))
        x = f.relu(self.block3_batch(x))
        x = self.block3_pool(x)

        # Block4
        x = f.relu(self.block4_conv1(x))
        x = f.relu(self.block4_conv2(x))
        x = f.relu(self.block4_conv3(x))
        x = f.relu(self.block4_conv4(x))
        x = f.relu(self.block4_batch(x))
        x = self.block4_pool(x)

        # Block5
        x = f.relu(self.block5_conv1(x))
        x = f.relu(self.block5_conv2(x))
        x = f.relu(self.block5_conv3(x))
        # シグモイド関数で出力
        x = torch.sigmoid(self.block5_conv4(x))
        x = torch.sigmoid(self.block5_batch(x))
        # x = self.block5_pool(x)

        return x


# 以下，動作確認
if __name__ == "__main__":
    # GPUで動かす
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net()
    model = model.to(device)

    # [バッチサイズ, チャンネル数, 縦,　横]をランダムに入力
    t = torch.rand([1, 3, 500, 332])
    t = t.to(device)
    output = model(t)
    print(output.size())








