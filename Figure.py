import numpy as np
import matplotlib.pyplot as plt

# 保存したデータの読み込みと表示
array_train = np.load('PascalVOC_train_loss_acc_backup_UNet_3.npz')
array_val = np.load("PascalVOC_val_loss_acc_backup_UNet_3.npz")
print('Train Loss:\n', array_train['loss'])
print('Train Accuracy:\n', array_train['acc'])
print("Validation Loss:\n", array_val["loss"])
print("Validation Accuracy\n", array_val["acc"])

# Lossのグラフの表示
plt.plot(array_train['loss'], color='blue', linewidth=1.5, label="Train")
plt.plot(array_val["loss"], color="orange", linewidth=1.5, label="Validation")
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0, 0.7)
plt.grid()
plt.legend(loc="upper right")
plt.show()
plt.savefig('Loss.png')

# Accuracyのグラフの表示
plt.plot(array_train['acc'], color='blue', linewidth=1.5, label="Train")
plt.plot(array_val["acc"], color="orange", linewidth=1.5, label="Validation")
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.grid()
plt.legend(loc="upper left")
plt.show()
plt.savefig('Accuracy.png')
