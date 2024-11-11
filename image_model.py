import torch
import pandas as pd
import numpy as np
import os
import pickle
from torch.utils.data import Dataset, DataLoader
from torch import nn
from d2l import torch as d2l
from torchmetrics import Accuracy, AUROC, F1Score
accuracy = Accuracy(task="binary")
auroc_metric = AUROC(task="binary")
f1_metric = F1Score(task="binary")
train_set = pd.read_csv("C:/Users/Azazel/Desktop/STAT3612/MLproject/stat-3612-group-project-2024-fall/train.csv")
test_set = pd.read_csv("C:/Users/Azazel/Desktop/STAT3612/MLproject/stat-3612-group-project-2024-fall/test.csv")
train_set['readmitted_within_30days'] = train_set['readmitted_within_30days'].replace({'True': 1, 'False': 0})
image_path = train_set['image_path']
image_label = train_set['readmitted_within_30days']
folder_images = os.listdir('C:/Users/Azazel/Desktop/STAT3612/MLproject/stat-3612-group-project-2024-fall/image_features/cxr_features')
matched_files = {}
print('inital work finished')
for path, label in zip(image_path, image_label):
    # 确保路径以 .pkl 结尾
    if not path.endswith('.pkl'):
        path = path + '.pkl'

    # 拼接完整路径
    full_path = os.path.join('C:/Users/Azazel/Desktop/STAT3612/MLproject/stat-3612-group-project-2024-fall/image_features/cxr_features', path)

    # 检查文件是否存在于 folder_images 中
    if path in folder_images:  # 直接在集合中查找
        matched_files[full_path] = label

print(f"Matched files: {len(matched_files)}")

def load_image_features(image_filename):
    with open(image_filename, 'rb') as f:
        image_features = pickle.load(f) 
    return torch.tensor(image_features, dtype=torch.float32)

class ReadmissionDataset(Dataset):
    def __init__(self, matched_files):
        self.matched_files = matched_files
        self.image_paths = list(matched_files.keys())
        self.labels = list(matched_files.values())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image_features = load_image_features(image_path)
        return image_features, torch.tensor(label, dtype=torch.float32)

dataset = ReadmissionDataset(matched_files)
train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

image_net = nn.Sequential(
    nn.Linear(1024, 512),
    nn.BatchNorm1d(512),
    nn.LeakyReLU(),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.Dropout(0.3),
    nn.LeakyReLU(),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.Dropout(0.3),
    nn.LeakyReLU(),
    nn.Linear(128, 1)
)


# 初始化权重
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)

image_net.apply(init_weights)

# 损失函数和优化器
loss = nn.BCEWithLogitsLoss()
trainer = torch.optim.AdamW(image_net.parameters(), lr=0.01, weight_decay=1e-4)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    image_net.train()  # 设置为训练模式
    running_loss = 0.0
    total_accuracy = 0.0
    total_auc = 0.0
    total_f1 = 0.0
    for i, (features, labels) in enumerate(train_loader):
        # 计算输出
        logits = image_net(features).squeeze()  # 得到 logits，形状为 [batch_size]
        batch_loss = loss(logits, labels)
        trainer.zero_grad()
        batch_loss.backward()
        trainer.step()
        
        running_loss += batch_loss.item()
        
        probabilities = torch.sigmoid(logits)
        batch_accuracy = accuracy(probabilities, labels)
        batch_auc = auroc_metric(probabilities, labels)
        batch_f1 = f1_metric(probabilities, labels)

        total_accuracy += batch_accuracy
        total_auc += batch_auc
        total_f1 += batch_f1

    # 输出每个 epoch 的平均 loss, accuracy 和 AUC
    avg_loss = running_loss / len(train_loader)
    avg_accuracy = total_accuracy / len(train_loader)
    avg_auc = total_auc / len(train_loader)
    avg_f1 = total_f1/len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}, Accuracy: {avg_accuracy}, AUC: {avg_auc}, F1:{avg_f1}")