import numpy as np
import cv2
import os
import glob
import torch
import torch.utils.data


# このデータセットは入力データとそれに対応するラベル1組を返すモジュール
class Mydataset(torch.utils.data.Dataset):
    def __init__(self, dir_original, dir_segmented, input_size, output_size, transform1=None, transform2=None):

        # このクラス内でずっと使う変数にはself.をつける
        self.transform1 = transform1
        self.transform2 = transform2
        self.input_size = input_size
        self.output_size = output_size

        self.paths_original, self.paths_segmented = self.generate_paths(dir_original, dir_segmented)

        self.data_num = len(self.paths_original)
        # print(self.data_num)

    # データセットに含まれる全要素の数を返す
    def __len__(self):
        return self.data_num

    # i番目のサンプルをdataset[i]という形で取得できるようにする
    def __getitem__(self, idx):

        original_path = str(self.paths_original[idx])
        # print(original_path)
        segmented_path = str(self.paths_segmented[idx])
        # print(segmented_path)

        color, depth_color = self.image_load(original_path, segmented_path, self.input_size, self.output_size)

        if self.transform1:
           color = self.transform1(original_path)
        if self.transform2:
           depth_color = self.transform2(segmented_path)

        return color, depth_color

    @staticmethod
    def generate_paths(dir_original, dir_segmented):
        # ファイル名を取得
        paths_original = glob.glob(dir_original + "/*")
        paths_segmented = glob.glob(dir_segmented + "/*")

        if len(paths_original) == 0 or len(paths_segmented) == 0:
            raise FileNotFoundError("Could not load images.")

        # 教師画像の拡張子を.jpgに書き換えたものが読み込むべき入力画像のファイル名になる
        filenames = list(map(lambda path: path.split(os.sep)[-1].split(".")[0], paths_segmented))
        paths_original = list(map(lambda filename: dir_original + "/" + filename + ".jpg", filenames))

        return paths_original, paths_segmented

    @staticmethod
    def image_load(original, segmented, input_size, output_size):

        # 画像読み込み，サイズ調整，配列
        original = cv2.imread(original)
        original = cv2.resize(original, (input_size, input_size))
        original = np.array(original, dtype=np.float32)
        segmented = cv2.imread(segmented, 0)
        # しきい値の設定
        threshold = 10
        # 二値化(しきい値10を超えた画素を255にする。)
        # cv2.threshold(画像, しきい値, しきい値を超えた場合に変更する値, 二値化の方法)
        ret, segmented = cv2.threshold(segmented, threshold, 255, cv2.THRESH_BINARY)
        segmented = cv2.resize(segmented, (output_size, output_size))
        segmented = np.array(segmented, dtype=np.float32)

        # 0~1に正規化
        original /= 255
        segmented /= 255

        # 次元の追加，[高さ，幅]から[高さ，幅,チャンネル数（1）]
        segmented = segmented[:, :, np.newaxis]

        # [height, width, channels]を[channels, height, width]に変換
        original = np.transpose(original, (2, 0, 1))
        segmented = np.transpose(segmented, (2, 0, 1))

        # numpy から torch tensorに変換
        original = torch.from_numpy(original)
        segmented = torch.from_numpy(segmented)

        return original, segmented


# 以下，動作確認
if __name__ == "__main__":

    dir_1 = "D:\\workspace\\programs\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\JPEGImages"
    dir_2 = "D:\\workspace\\programs\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\SegmentationClass"

    dataset = Mydataset(dir_1, dir_2, 448, 14, transform1=None, transform2=None)
    print(len(dataset))
    # 2913枚
    # print(dataset[3])

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)
    for step, (original, segmented) in enumerate(trainloader, 1):
        print(original.size())
        print(segmented.size())











