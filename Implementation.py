import numpy as np
import cv2
import torch
from VGG19_model import Net


# モデルの表示
model = Net()
print(model)
# モデルをGPUに対応させる
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# 学習済みモデルの読み込み
model_path = 'PascalVOC_model_2.pth'
model.load_state_dict(torch.load(model_path))

# __入力画像の処理__
# 画像の読み込み，表示
input_image = cv2.imread("D:\\workspace\\programs\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\JPEGImages\\2007_000003.jpg")
cv2.imshow('Input Image', input_image)
cv2.waitKey(0)
print(input_image.shape)
# 画像サイズの変換
input_image = cv2.resize(input_image, (448, 448))
print(input_image.shape)
# float32型に変換
input_image = np.array(input_image, dtype=np.float32)
# 0~1に正規化
input_image /= 255
# [高さ，幅，チャンネル数]から[チャンネル数，高さ，幅]に変換
input_image = np.transpose(input_image, (2, 0, 1))
print(input_image.shape)
# 次元の追加，[チャンネル数，高さ，幅]から[バッチ数（１），チャンネル数，高さ，幅]
input_image = input_image[np.newaxis, :, :, :]
print(input_image.shape)
# numpy から torch tensor　に変換
input_image = torch.from_numpy(input_image)
print(input_image.size())
# データをGPUに送る
input_image = input_image.to(device)

# データをモデルに送る
output_image = model(input_image)
print(output_image.shape)

# __出力画像の処理__
# torch tensor から numpy　に変換
output_image = output_image.to('cpu').detach().numpy()
# [バッチ数，チャンネル数，高さ，幅]からバッチ数を削除
output_image = np.squeeze(output_image)
print(output_image.shape)
# 画像サイズの変換
output_image = cv2.resize(output_image, (1355, 789), interpolation=cv2.INTER_NEAREST)
# 画像の表示
cv2.imshow('Output Image', output_image)
cv2.waitKey(0)
# 画像の保存
output_image = np.array(output_image, dtype=np.uint8)
image_path = "D:\\workspace\\programs\\sample\\.idea\\picture\\"
cv2.imwrite(image_path + "test_1_out.png", output_image)




