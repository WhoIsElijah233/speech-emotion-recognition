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
from torch.utils.data import DataLoader
from dataset import EmotionDataset
from model import EmotionCNN


# 1 数据集
dataset = EmotionDataset()

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)


# 2 模型
device = "cuda" if torch.cuda.is_available() else "cpu"

model = EmotionCNN().to(device)


# 3 loss 和 optimizer
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# 4 训练
epochs = 2

for epoch in range(epochs):

    total_loss = 0

    for x, y in loader:

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

    print("epoch:", epoch, "loss:", total_loss / len(loader))
