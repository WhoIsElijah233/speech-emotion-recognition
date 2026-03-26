# # training script
# import torch
# from torch.utils.data import DataLoader

# from dataset import EmotionDataset
# from model import EmotionModel

# dataset = EmotionDataset()
# loader = DataLoader(dataset, batch_size = 4, shuffle = True)

# model = EmotionModel()
# loss_fn = torch.nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(
#   model.parameters(),
#   lr = 0.001
# )

# for epoch in range(3):
#   print("epoch:", epoch)

#   for audio, label in loader:
#     pred = model(audio)
#     loss = loss_fn(pred, label)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     print("loss:", loss.item())

import torch
from torch.utils.data import DataLoader, random_split
from dataset import EmotionDataset
from model import EmotionCNN
import matplotlib.pyplot as plt


# 1 数据集
dataset = EmotionDataset()

# 划分训练集、验证集、测试集
train_size = int(0.7*len(dataset))
val_size = int(0.15*len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print("dataloader ready")

# loader = DataLoader(
#     dataset,
#     batch_size=16,
#     shuffle=True
# )


# 2 模型
device = "cuda" if torch.cuda.is_available() else "cpu"

model = EmotionCNN().to(device)
print("model ready")

# 3 loss 和 optimizer
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# 4 训练
epochs = 50
train_losses = []
val_accuracies = []
best_val_acc = 0

def calculate_accuracy(loader):
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
    for x, y in loader:
      x, y = x.to(device), y.to(device)
      pred = model(x)
      _, predicted = torch.max(pred, 1)  # 取预测的类别
      total += y.size(0)
      correct += (predicted==y).sum().item()
  model.train()
  return correct/total

print("start training. . .")
for epoch in range(epochs):

  total_loss = 0

  for x, y in train_loader:

    x = x.to(device)
    y = y.to(device)

    # forward
    pred = model(x)

    loss = criterion(pred, y)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
  avg_loss = total_loss/len(train_loader)
  train_losses.append(avg_loss)

  val_acc = calculate_accuracy(val_loader)
  val_accuracies.append(val_acc)

  if val_acc > best_val_acc:
    best_val_acc = val_acc
    torch.save(model.state_dict(), 'best_model.pth')
    print(f"Epoch {epoch + 1}: model saved. accuracy: {best_val_acc:.4f}")

  test_acc = calculate_accuracy(test_loader)  # 你需要定义这个

  print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, val_acc: {val_acc:.4f}, test_acc: {test_acc:.4f}")


# 6 画图
plt.figure(figsize=(12, 4))

# 左图：损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, 'b-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('training loss')
plt.grid(True)

# 右图：验证准确率
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), val_accuracies, 'r-')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('val accuracy')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"\nresult report:")
print(f"test acc: {test_acc*100:.2f}%")
print(f"best val acc: {best_val_acc*100:.2f}%")
