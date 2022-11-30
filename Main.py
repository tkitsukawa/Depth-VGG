# base
import numpy as np
# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
# from import
from PascalVOC_Dataset import Mydataset
from VGG19_model import Net


# 学習用関数
# Trainデータ
def train(train_loader):
    # 訓練モードに設定
    model.train()
    loss_total = 0.0
    acc_total = 0

    for i, (colors, depth_colors) in enumerate(train_loader, 1):

        # データをGPUに送る
        colors = colors.to(device)
        depth_colors = depth_colors.to(device)

        # 勾配の初期化
        optimizer.zero_grad()
        # 出力計算
        outputs = model(colors)
        # 出力結果と正解ラベルから損失計算
        loss = criterion(outputs, depth_colors)
        # 勾配の計算
        loss.backward()
        # パラメータの更新
        optimizer.step()
        # 誤差の保存
        loss_total += loss.item()

        # 正答率の算出
        acc = torch.abs(outputs.detach() - depth_colors.detach())
        # print(acc.size())
        acc_correct = acc[acc < 0.1]
        # print(acc_correct.size())
        acc_total += acc_correct.size(0) / (acc.size(0) * acc.size(1) * acc.size(2) * acc.size(3))

    train_loss = loss_total / i
    train_acc = acc_total / i

    return train_loss, train_acc


# 検証用関数
# Validationデータ
def eval(val_loader):
    # 評価モードに設定
    model.eval()
    loss_total = 0.0
    acc_total = 0

    # 学習させない
    with torch.no_grad():
        # iが１進むとcolorsとdepth_colorsが１進む
        for i, (colors, depth_colors) in enumerate(val_loader, 1):

            # データをGPUに送る
            colors = colors.to(device)
            depth_colors = depth_colors.to(device)

            # 出力計算
            outputs = model(colors)
            # 出力結果と正解ラベルから損失計算
            loss = criterion(outputs, depth_colors)
            # 誤差の保存
            loss_total += loss.item()

            # 正答率の算出
            acc = torch.abs(outputs.detach() - depth_colors.detach())
            # print(acc.size())
            acc_correct = acc[acc < 0.1]
            # print(acc_correct.size())
            acc_total += acc_correct.size(0) / (acc.size(0) * acc.size(1) * acc.size(2) * acc.size(3))

        val_loss = loss_total / i
        val_acc = acc_total / i

        return val_loss, val_acc


if __name__ == "__main__":

    # ハイパーパラメータの設定
    epochs = 100
    batch_size = 1
    learning_rate = 0.001

    # 画像サイズの設定
    input_size = 448
    output_size = 28

    # GPUに対応させるための設定（ここ３行はこのまま)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net()
    model = model.to(device)
    print(model)

    # Optimizer(パラメータの更新)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 損失関数の設定
    criterion = nn.BCELoss()

    # グラフ用の配列の準備
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    # データセットオブジェクト
    dir_original = "D:\\workspace\\programs\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\JPEGImages"
    dir_segmented = "D:\\workspace\\programs\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\SegmentationClass"
    full_dataset = Mydataset(dir_original, dir_segmented, input_size=input_size, output_size=output_size, transform1=None, transform2=None)

    # データセットを学習用と検証用に分割
    train_dataset_length = int(len(full_dataset) * 0.9)
    val_dataset_length = int(len(full_dataset)) - train_dataset_length
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_dataset_length, val_dataset_length])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=4)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # エポック数だけ学習
    for epoch in range(epochs):

        train_loss, train_acc = train(train_loader)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        val_loss, val_acc = eval(val_loader)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        print("epoch %d, train_loss: %.4f, train_accuracy: %.4f, validation_loss: %.4f, validation_accuracy: %.4f" %
              (epoch + 1, train_loss, train_acc, val_loss, val_acc))

    # モデルとグラフの保存
    np.savez("PascalVOC_train_loss_acc_backup_2.npz", loss=np.array(train_loss_list), acc=np.array(train_acc_list))
    np.savez("PascalVOC_val_loss_acc_backup_2.npz", loss=np.array(val_loss_list), acc=np.array(val_acc_list))
    torch.save(model.state_dict(), "PascalVOC_model_2.pth")

    print("[Finished Training]")

